import unittest
from util.math import log_normalize, scale

class TestLists(unittest.TestCase):

    def test_log_normalize(self):
        self.assertArrayAlmostEqual(log_normalize([], None), [])
        self.assertArrayAlmostEqual(log_normalize([1], 1.0), [1])

        # A temperature of 1.0 should be equivalent to standard normalization
        self.assertArrayAlmostEqual(log_normalize([1, 1], 1.0), [1/2, 1/2])
        self.assertArrayAlmostEqual(log_normalize([1, 2, 3], 1.0), [1/6, 2/6, 3/6])

        # As temperature approaches zero, highest value should get all weight
        self.assertArrayAlmostEqual(log_normalize([1, 2, 3], 0.01), [0, 0, 1])

        # As temperature approaches infinity, weights should become uniform
        self.assertArrayAlmostEqual(log_normalize([1, 2, 3], 10000000.0), [1/3, 1/3, 1/3])

    def test_scale(self):
        self.assertEqual(scale([0.5], 10), [10])
        self.assertEqual(scale([0.5, 0.5], 10), [5, 5])
        self.assertEqual(scale([0.0, 0.5], 10), [0, 10])
        self.assertEqual(scale([0.0, 0.5], 10, 1), [1, 9])
        self.assertEqual(scale([0.0, 0.2, 0.8], 100, 1), [1, 20, 79])

    def assertArrayAlmostEqual(self, xs, ys):
        self.assertEqual(len(xs), len(ys))
        for x, y in zip(xs, ys):
            self.assertAlmostEqual(x, y)

