import unittest
from random import choice
from .packing import BYTES_IN_INT
from .padding import pad, unpad
from model import Model
from test.mock_keras import mock_keras

class TestModeling(unittest.TestCase):

    def test_padding(self):
        """ Test round-tripping padding.

        Note: This is a non-deterministic test, but should always pass.
        """
        model = mock_keras()

        # Blocksizes that aren't a multiple of BYTES_IN_INT should error.
        for i in range(BYTES_IN_INT):
            with self.assertRaises(ValueError):
                pad(model, [], "", i)

        # Padding should always add a boundary character if it doesn't end in one
        for message in map(list, ["", "1", "12", "012", "102"]):
            padded = pad(model, [], message, BYTES_IN_INT)
            unpadded = unpad(model, padded)
            self.assertEqual(unpadded, message + ['0'])

        # Padding should not add a boundary character if it already ends in one
        for message in map(list, ["0", "00", "10", "120", "0120", "1020"]):
            padded = pad(model, [], message, BYTES_IN_INT)
            unpadded = unpad(model, padded)
            self.assertEqual(unpadded, message)

        # Test various blocksizes and message lengths
        blocksizes = range(BYTES_IN_INT, 10 * BYTES_IN_INT, BYTES_IN_INT)

        for message_length in range(0, 20):
            for blocksize in blocksizes:
                message = [choice("012") for _ in range(message_length)] + ['0']
                padded = pad(model, [], message, blocksize)
                self.assertEqual((len(padded) * BYTES_IN_INT) % blocksize, 0)
                self.assertEqual(message, list(unpad(model, padded)))

    def test_unpad(self):
        model = mock_keras()

        self.assertEqual(unpad(model, ""), "")
        self.assertEqual(unpad(model, "0"), "")
        self.assertEqual(unpad(model, "00"), "0")
        self.assertEqual(unpad(model, "10"), "1")
        self.assertEqual(unpad(model, "010"), "0")
        self.assertEqual(unpad(model, "110"), "11")
        self.assertEqual(unpad(model, "0110"), "0")

