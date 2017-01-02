import unittest
from random import choice
from encoding import encode, decode
from mock_model import mock_model

class TestEncoding(unittest.TestCase):

    def test_encoding(self):
        """ Test round-trip encoding.

        Note: This is a non-deterministic test, but should always pass.
        """
        model = mock_model()

        # If message doesn't end in boundary, it should be appended.
        self.assertEqual(decode(model, encode(model, "")), "0")
        self.assertEqual(decode(model, encode(model, "0")), "0")
        self.assertEqual(decode(model, encode(model, "1")), "10")

        # Test a bunch of random messges of varying lengths
        for i in range(30):
            message = "".join(choice("01") for _ in range(i)) + model.config.model.boundary
            result = decode(model, encode(model, message))
            self.assertEqual(message, result)
