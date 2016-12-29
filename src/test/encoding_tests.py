import unittest
from random import choice
from encoding import encode, decode
from test.mock_keras import mock_keras

config = {
    'alphabet': '01',
    'nodes': 0,
    'sequence_length': 1,
    'normalizing_length': 10,
    'priming_length': 10,
    'max_padding_trials': 100,
    'boundary': '1'
}

class TestEncoding(unittest.TestCase):

    def test_encoding(self):
        """ Test round-trip encoding.

        Note: This is a non-deterministic test, but should always pass.
        """
        model = mock_keras(config)

        # If message doesn't end in boundary, it should be appended.
        self.assertEqual(decode(model, encode(model, "")), "1")
        self.assertEqual(decode(model, encode(model, "0")), "01")
        self.assertEqual(decode(model, encode(model, "1")), "1")

        # Test a bunch of random messges of varying lengths
        for i in range(30):
            message = "".join(choice("01") for _ in range(i)) + "1" # Ends in boundary
            result = decode(model, encode(model, message))
            self.assertEqual(message, result)
