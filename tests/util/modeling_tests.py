import unittest
from random import choice
from util.modeling import tabulate, recite
from mock_model import config, mock_model

class TestModeling(unittest.TestCase):

    def test_modeling(self):
        """ Test round-tripping tabulate / recite.

        Note: This is a non-deterministic test, but should always pass.
        """
        model = mock_model()

        for i in range(20):
            alphabet = model.config.model.alphabet
            message = [choice(alphabet) for _ in range(i)]
            result = recite(model, [], tabulate(model, [], message))
            self.assertEqual(message, list(result))
