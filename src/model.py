from os.path import join
import re
import json
import numpy as np
from util.keras import model_from_json
from util.one_hot_encoding import one_hot_encoding
from util.math import log_normalize

class Model(object):
    """Contains the Keras model and related configuration values to
    encode / decode values.

    Here is a description of the configuration map:

        alphabet (string): A string containing the entire alphabet of the
            model.

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
        model (Model): The Keras model that has been trained on a given domain.

        config (dict): A config object containing various parameters related to
            prediction. See above for description.

    Attrs:
        model (Keras Model): The Keras model.

        alphabet (string): The string of all characters in the model's alphabet.

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

    def __init__(self, model, config):
        self.model = model
        self.alphabet = sorted(config['alphabet'])
        self.normalizing_length = config['normalizing_length']
        self.priming_length = config['priming_length']
        self.max_padding_trials = config.get('max_padding_trials', 1000)
        self.padding_novelty_growth_rate = config.get('padding_novelty_growth_rate', 1.01)
        self.boundary = config['boundary']
        self.novelty = config.get('novelty', 0.4)
        self._transformations = config.get('transformations', {})
        self.sequence_length = model.input_shape[1]
        self._validate()

    def _validate(self):
        """ Validates the config values provided to initialization. """
        if len(self.alphabet) != self.model.input_shape[2]:
            raise Exception("Model expects an alphabet of length %s, but the config provided an alphabet of length %s." % (self.model.input_shape[2], len(self.alphabet)))

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

def load_model(directory):
    """Loads a model from a given directory.

    The directory should contain three files:
        1) model.json - output from the Keras model.
        2) model.weights - output from the Keras model.
        3) config.json - config describing metadata about the model.

    Args:
        directory (string): The directory containing the files to load the model

    Returns:
        The loaded model.

    Raises:
        Exception: If the model config fails to validate.
    """
    with open(join(directory, 'model.json')) as model_file:
        model = model_from_json(model_file.read())
    model.load_weights(join(directory, 'model.weights'))

    with open(join(directory, 'config.json')) as config_file:
        config = json.load(config_file)

    return Model(model, config)
