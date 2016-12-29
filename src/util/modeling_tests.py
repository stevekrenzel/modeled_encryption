import unittest
from random import choice
from .modeling import tabulate, recite
from test.mock_keras import mock_keras

config = {
    'alphabet': '01',
    'nodes': 0,
    'sequence_length': 5,
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
        model = mock_keras(config)

        for i in range(20):
            message = [choice("01") for _ in range(i)]
            result = recite(model, [], tabulate(model, [], message))
            self.assertEqual(message, list(result))
