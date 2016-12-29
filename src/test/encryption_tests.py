import unittest
from random import choice
from encryption import encrypt, decrypt
from test.mock_keras import mock_keras

config = {
    'alphabet': '01',
    'nodes': 0,
    'sequence_length': 3,
    'normalizing_length': 10,
    'priming_length': 10,
    'max_padding_trials': 100,
    'boundary': '1'
}

class TestEncryption(unittest.TestCase):

    def test_encryption(self):
        """ Test round-trip encryption.

        Note: This is a non-deterministic test, but should always pass.
        """
        model = mock_keras(config)

        # If message doesn't end in boundary, it should be appended.
        self.assertEqual(decrypt(model, "foo", encrypt(model, "foo", "")), "1")
        self.assertEqual(decrypt(model, "foo", encrypt(model, "foo", "0")), "01")
        self.assertEqual(decrypt(model, "foo", encrypt(model, "foo", "1")), "1")

        # Test a bunch of random messges of varying lengths
        for i in range(30):
            message = "".join(choice("01") for _ in range(i)) + "1" # Ends in boundary
            result = decrypt(model, "foo", encrypt(model, "foo", message))
            self.assertEqual(message, result)

        # Test a bunch of random messges using wrong key
        for i in range(60, 90): # Short bit strings may match by chance, so we go long
            message = "".join(choice("01") for _ in range(i)) + "1" # Ends in boundary
            result = decrypt(model, "bar", encrypt(model, "foo", message))
            self.assertNotEqual(message, result)