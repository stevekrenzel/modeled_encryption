import unittest
from util.one_hot_encoding import one_hot_encoding

def ohe(xs, classes):
    return one_hot_encoding(xs, classes).tolist()

class TestOneHotEncoding(unittest.TestCase):

    def test_one_hot_encoding(self):
        self.assertEqual(ohe([], range(3)), [])
        self.assertEqual(ohe([0], range(1)), [[True]])
        self.assertEqual(ohe("abc", "abcd"), [[True, False, False, False], [False, True, False, False], [False, False, True, False]])

        # Test out of bounds value
        with self.assertRaises(ValueError):
            ohe([1], range(1))
