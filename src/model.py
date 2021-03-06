import re
import numpy as np
from os.path import isfile
from functools import partial
from random import choice
from config import load_config
from util.keras import Sequential, LSTM, Dense, Activation
from util.one_hot_encoding import one_hot_encoding
from util.math import log_normalize
from util.modeling import recite
from util.randoms import random_ints

class Model(object):
    """ A model that can learn and predict sequences.

    Attrs:
        config (Config): The model's config.
    """

    def __init__(self, config):
        """ Instantiates a model instance given a config object.

        Args:
            config (Config): The model's config object.

        Raises:
            Exception: If model fails to build.
        """
        self.config = config
        self.model = self._create_model()

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
            novelty = self.config.encoding.novelty

        alphabet = self.config.model.alphabet
        encoded = one_hot_encoding(sequence, alphabet)
        nested = np.array([encoded], dtype=np.bool)
        probabilities = self.model.predict(nested, verbose=0)[0]
        return log_normalize(probabilities, novelty)

    def train(self, data):
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
        alphabet = self.config.model.alphabet
        sequence_length = self.config.model.sequence_length
        batch_size = self.config.training.batch_size
        epochs = self.config.training.epochs
        validation_split = self.config.training.validation_split
        weights_file = self.config.model.weights_file

        self.model.summary()
        transformed = self.transform(data)
        encoded = one_hot_encoding(transformed, alphabet)
        X = np.array([encoded[i : i + sequence_length] for i in range(len(encoded) - sequence_length)])
        y = np.array(encoded[sequence_length:])
        for i in range(epochs):
            print()
            print("-" * 79)
            print("Epoch %s" % (i))
            self.model.fit(X, y, validation_split=validation_split, batch_size=batch_size, nb_epoch=1, shuffle=True)
            self.model.save(weights_file)
            print("Saved weights to '%s'" % (weights_file))
            print("Sampling model: ")
            print(self.sample(50))

    def sample(self, size, novelty=None):
        """ Generates sample output from the model.

        Example:
            >> model.sample(10)
            'OPERATE IN THE AREAS'

        Args:
            size (int): The length of output to generate.

            novelty (optional: float): The novelty to use when generating the sequence.

        Returns:
            A sequence of characters generated by the model.
        """
        alphabet = self.config.model.alphabet
        sequence_length = self.config.model.sequence_length

        initial = [choice(alphabet) for _ in range(sequence_length - 1)] + [self.config.model.boundary]
        sequence = recite(self, initial, random_ints(), novelty)
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
        alphabet = self.config.model.alphabet
        transformations = self.config.transformations
        translate = transformations.translate
        substitutions = transformations.substitutions

        if translate != None:
            translate = transformations.translate
            original = translate[0]
            translated = translate[1]
            data = data.translate(str.maketrans(original, translated))

        if substitutions != None:
            for (pattern, sub) in substitutions:
                regex = re.compile(pattern)
                data = regex.sub(sub, data)

        chars = set(alphabet)
        if any(c not in chars for c in data):
            raise Exception("Data contains non-alphabet characters post-transformation. Can't continue.")

        return data

    def _create_model(self):
        alphabet = self.config.model.alphabet
        sequence_length = self.config.model.sequence_length
        nodes = self.config.model.nodes
        weights_file = self.config.model.weights_file

        alphabet_size = len(alphabet)
        input_shape = (sequence_length, alphabet_size)
        loss = 'categorical_crossentropy'
        optimizer = 'adadelta'
        metrics = ['accuracy']

        hidden_layer = LSTM(nodes,
                            input_shape=input_shape,
                            consume_less="cpu")
        output_layer = Dense(alphabet_size)
        activation = Activation('softmax')

        model = Sequential()
        model.add(hidden_layer)
        model.add(output_layer)
        model.add(activation)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        if isfile(weights_file):
            model.load_weights(weights_file)

        return model

def load_model(config_file):
    """Loads a model from a given config file.

    Args:
        config_file (string): The filename of the config file to load the model from.

    Returns:
        The loaded model.

    Raises:
        Exception: If the model config fails to validate.

        Exception: If the keras model fails to build.
    """
    config = load_config(config_file)
    return Model(config)
