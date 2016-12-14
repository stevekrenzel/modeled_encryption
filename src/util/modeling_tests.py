import unittest
from random import choice
from .modeling import tabulate, recite
from model import Model

class MockKerasModel(object):
    """ A mock keras model with sequence_length of 5, alphabet of 2 characters,
    and always predicts each character with equal probability.
    """

    def __init__(self):
        self.input_shape = (0, 5, 2)

    def predict(self, sequence, verbose):
        return [[1/2, 1/2]]

config = {
    'alphabet': '01',
    'normalizing_length': 0,
    'priming_length': 0,
    'max_padding_trials': 0,
    'boundary': '1'
}

class TestModeling(unittest.TestCase):

    def test_modeling(self):
        """ Test round-tripping tabulate / recite.

        Note: This is a non-deterministic test, but should always pass.
        """
        model = Model(MockKerasModel(), config)

        for i in range(20):
            message = [choice("01") for _ in range(i)]
            result = recite(model, [], tabulate(model, [], message))
            self.assertEqual(message, list(result))
