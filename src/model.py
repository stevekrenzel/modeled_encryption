from os.path import join, isfile
import re
import json
import numpy as np
from functools import partial
from random import choice
from util.keras import Sequential, LSTM, Dense, Activation
from util.one_hot_encoding import one_hot_encoding
from util.math import log_normalize
from util.modeling import recite
from util.randoms import random_ints

class Model(object):
    """Contains the Keras model and related configuration values to
    encode / decode values.

    Here is a description of the configuration map:

        alphabet (string): A string containing the entire alphabet of the
            model.

        nodes (int): The number of nodes in the hidden layer of the the LSTM.

        sequence_length (int): The length of the input sequence to the model.

        normalizing_length (int): The number of characters to run through a
            randomized model to normalize it's output distribution..

        priming_length (int): The number of characters to generate to
            create an initial sequence for encoding.

        max_padding_trials (int: 1000, optional): An integer containing the
            maximum number of times to try adding padding before it gives up.

        padding_novelty_growth_rate (int: 1.01, optional): A float represnting
            the increase in novelty after each failed attempt at padding. We
            increase the novelty slightly after each failure in an attempt to
            increase the odds of generating padding of a valid length while
            minimizing the skew to the model's output distribution.

        boundary (char): The delimiting character between tokens in the model.
            (e.g. ' ' for a model that generates words, or '.' for a model that
            generates sentences)

        novelty (float: 0.4, optional): The novelty (a.k.a. temperature) to use
            when normalizing prediction weights. A smaller number will make
            the predictions more conservative.

        transformations (dict, optional): A dictionary describing how to
            transform data before running it through the model.

            Translations are ran before substitutions. Substitutions are ran in
            the order that they are defined.

            translate ((string, string), optional): A tuple with two strings of
                same length. Characters from the first string will be replaced
                with corresponding characters from the second string.

            substitutions ([(regex, string)], optional): A list of regular
                expressions and the strings that should replace their matches.

    Args:
        config (dict): A config object containing various parameters related to
            prediction. See above for description.

        model_builder (self -> keras model): A function that returns an
        instantiated keras model.

    Attrs:
        model (Keras Model): The Keras model.

        alphabet (string): The string of all characters in the model's alphabet.

        nodes (int): The number of nodes in the hidden layer of the the LSTM.

        normalizing_length (int): The number of characters to generate when
            normalizing the model.

        priming_length (int): The number of characters to generate when priming
            the model.

        max_padding_trials (int): The maximum number of attempts to generate
            padding during the encoding process before giving up.

        padding_novelty_growth_rate (float): The rate to grow the default
            novelty when generating padding.

        boundary (char): The character the seperates tokens in the output of
            the model.

        novelty (float): The novelty rate used when normalizing prediction
            probabilities.

        sequence_length (int): The length of the input sequence to the model.

    Raises:
        Exception: If the alphabet is the wrong length.

        Exception: If the boundary isn't present in the alphabet.
    """

    def __init__(self, config, model_builder):
        self.alphabet = sorted(config['alphabet'])
        self.nodes = config['nodes']
        self.sequence_length = config['sequence_length']
        self.normalizing_length = config['normalizing_length']
        self.priming_length = config['priming_length']
        self.max_padding_trials = config.get('max_padding_trials', 1000)
        self.padding_novelty_growth_rate = config.get('padding_novelty_growth_rate', 1.01)
        self.boundary = config['boundary']
        self.novelty = config.get('novelty', 0.4)
        self._transformations = config.get('transformations', {})
        self._validate()
        self.model = model_builder(self)

    def _validate(self):
        """ Validates the config values provided to initialization. """
        if self.boundary not in self.alphabet:
            raise Exception("The boundary must be preset in the alphabet.")

    def predict(self, sequence, novelty=None):
        """ Given a sequence, returns the probabilities of each character in
        the alphabet following the sequence.

        Note: A large novelty will cause the probabilities to converge to a
        uniform distribution.

        Example:
            >> phrase = 'THIS IS A VERY LONG EXAMPLE SEQUENCE TO BE USED IN'
            >> probs = model.predict(phrase)
            >> [round(p, 2) for p in probs] # Round probabilities for simplicity
            [0.0, 0.0, 0.83, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            >> list(zip(model.alphabet, [round(p, 2) for p in probs]))
            [(' ', 0.83), <... ommitted ...>, ('T', 0.17), <... ommitted ...>]

            >> novelty = 100.0 # Huge for a novelty, will make values uniform-ish
            >> probs = model.predict(phrase, novelty)
            >> [round(p, 2) for p in probs]
            [0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.03, 0.03]

        Args:
            sequence (string): The sequence of characters to predict the next
                character for.

            novelty (float): The conservativeness of the prediction. A smaller
                number results in more conservative estimates.

        Returns:
            A list of probabilities corresponding to the liklihood of each
            letter in the model's alphabet occuring next in the sequence.

        Raises:
            ValueError: If `sequence` contains an item that is not present in
            the model's alphabet.
        """
        if novelty == None:
            novelty = self.novelty

        encoded = one_hot_encoding(sequence, self.alphabet)
        nested = np.array([encoded], dtype=np.bool)
        probabilities = self.model.predict(nested, verbose=0)[0]
        return log_normalize(probabilities, novelty)

    def train(self, data, batch_size=256, epochs=200, validation_split=0.05):
        """ Trains the model on the provided data.

        This will print out a summary of the model structure, followed by
        metrics and progress on training. After each epoch, the weights will
        be saved to the model's directory, if one is provided.

        Args:
            data (string): The data to train the model on.

            batch_size (int): The batch size for training.

            epochs (int): The number of times to train on the entire data set.

            validation_split (float): The percentage of data to use for validation.

        Returns:
            Nothing. Updates the internal state of the model.
        """
        self.model.summary()
        transformed = self.transform(data)
        encoded = one_hot_encoding(transformed, self.alphabet)
        X = np.array([encoded[i : i + self.sequence_length] for i in range(len(encoded) - self.sequence_length)])
        y = np.array(encoded[self.sequence_length:])
        for i in range(epochs):
            print()
            print("-" * 79)
            print("Epoch %s" % (i))
            self.model.fit(X, y, validation_split=validation_split, batch_size=batch_size, nb_epoch=1, shuffle=True)
            #if self.directory != None:
            #    pass
            #    #self.model.save(join(self.directory, 'model.weights'))

    def sample(self, size):
        """ Generates sample output from the model. """
        initial = [choice(self.alphabet) for _ in range(self.sequence_length)]
        sequence = recite(self, initial, random_ints())
        return "".join(c for (c, _) in zip(sequence, range(size)))

    def transform(self, data):
        """Applies the model's transformations to the supplied data.

        Example:
            >> model.transform("Hello, my name is Steve.")
            'HELLO MY NAME IS STEVE '

        Args:
            data (string): The data to transform.

        Returns:
            The transformed data.

        Raises:
            Exception: If data contains characters that aren't specified in the
            alphabet, after all of the transformations are performed.
        """
        if 'translate' in self._transformations:
            translate = self._transformations['translate']
            original = translate[0]
            translated = translate[1]
            data = data.translate(str.maketrans(original, translated))

        if 'substitutions' in self._transformations:
            for (pattern, sub) in self._transformations['substitutions']:
                regex = re.compile(pattern)
                data = regex.sub(sub, data)

        chars = set(self.alphabet)
        if any(c not in chars for c in data):
            raise Exception("Data contains non-alphabet characters post-transformation. Can't continue.")

        return data

def create_keras_model(directory, load_weights, base):
    alphabet_size = len(base.alphabet)

    input_shape = (base.sequence_length, alphabet_size)
    loss = 'categorical_crossentropy'
    optimizer = 'adadelta'
    metrics = ['accuracy']

    hidden_layer = LSTM(base.nodes,
                        input_shape=input_shape,
                        consume_less="cpu")
    output_layer = Dense(alphabet_size)
    activation = Activation('softmax')

    model = Sequential()
    model.add(hidden_layer)
    model.add(output_layer)
    model.add(activation)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    weights_path = join(directory, 'model.weights')
    if load_weights and isfile(weights_path):
        model.load_weights(weights_path)

    return model

def load_model(directory, load_weights=True):
    """Loads a model from a given directory.

    The directory should contain two files:
        1) model.weights - output from the Keras model.
        2) config.json - config describing metadata about the model.

    Args:
        directory (string): The directory containing the files to load the model
        load_weights (bool): Whether or not to use the existing weights file

    Returns:
        The loaded model.

    Raises:
        Exception: If the model config fails to validate.
    """
    with open(join(directory, 'config.json')) as config_file:
        config = json.load(config_file)

    builder = partial(create_keras_model, directory, load_weights)
    model = Model(config, builder)
    return model
