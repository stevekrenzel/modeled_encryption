import unittest
from .sampling import choose_choice, choose_weight

class TestSampling(unittest.TestCase):

    def test_sampling(self):
        """ This is a non-deterministic round-trip test. """
        for _ in range(10):
            choices = [1, 2, 3, 4, 5]
            weights = [1, 2, 3, 4, 5]
            for choice in choices:
                weight = choose_weight(choice, choices, weights)
                chosen = choose_choice(weight, choices, weights)
                self.assertEqual(choice, chosen)

    def test_choose_choice(self):
        # No negative weights
        with self.assertRaises(ValueError):
            choose_choice(-1, [], [])

        # Weights and choices must be equal length
        with self.assertRaises(ValueError):
            choose_choice(0, [0], [])

        # Weights must sum to greater than weight
        with self.assertRaises(ValueError):
            choose_choice(2, [1, 0], [0, 1])

        # Can't have zero choices
        with self.assertRaises(ValueError):
            choose_choice(0, [], [])

        self.assertEqual(choose_choice(0, "a", [1]), "a")

        # Single weight
        self.assertEqual(choose_choice(0, "a", [2]), "a")
        self.assertEqual(choose_choice(1, "a", [2]), "a")

        # Uniform weight
        self.assertEqual(choose_choice(0, "ab", [1, 1]), "a")
        self.assertEqual(choose_choice(1, "ab", [1, 1]), "b")

        # Skewed weight
        self.assertEqual(choose_choice(0, "ab", [1, 2]), "a")
        self.assertEqual(choose_choice(1, "ab", [1, 2]), "b")
        self.assertEqual(choose_choice(2, "ab", [1, 2]), "b")

        # Zero weight
        self.assertEqual(choose_choice(0, "abc", [1, 0, 2]), "a")
        self.assertEqual(choose_choice(1, "abc", [1, 0, 2]), "c")
        self.assertEqual(choose_choice(2, "abc", [1, 0, 2]), "c")

    def test_choose_weight(self):
        # Weights and choices must be equal length
        with self.assertRaises(ValueError):
            choose_weight("a", "a", [])

        # Choice must be present in choices
        with self.assertRaises(ValueError):
            choose_weight("b", "a", [0])

        # Single choice
        self.assertEqual(choose_weight("a", "a", [1]), 0)

        # Uniform determinisitic
        self.assertEqual(choose_weight("a", "ab", [1,1]), 0)
        self.assertEqual(choose_weight("b", "ab", [1,1]), 1)

        # Non-deterministic sampling here
        self.assertTrue(choose_weight("a", "abc", [2,2,2]) in [0, 1])
        self.assertTrue(choose_weight("a", "abc", [2,2,2]) in [0, 1])
        self.assertTrue(choose_weight("b", "abc", [2,2,2]) in [2, 3])
        self.assertTrue(choose_weight("b", "abc", [2,2,2]) in [2, 3])
        self.assertTrue(choose_weight("c", "abc", [2,2,2]) in [4, 5])
        self.assertTrue(choose_weight("c", "abc", [2,2,2]) in [4, 5])
