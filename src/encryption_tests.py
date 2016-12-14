import unittest
from random import choice
from encryption import encrypt, decrypt
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
        model = Model(MockKerasModel(), config)

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
