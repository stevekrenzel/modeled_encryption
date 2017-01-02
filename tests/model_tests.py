import unittest
from random import choice
from encoding import encode, decode
from mock_model import mock_model, config

class TestModel(unittest.TestCase):

    def test_predict(self):
        cfg = config()
        cfg['model']['alphabet'] = "01"
        model = mock_model(cfg)
        sequence = "001"
        result = model.predict(sequence)

        # Ensure we're one-hot encoding
        self.assertEqual(model.model.last_sequence.tolist(), [[[True, False], [True, False], [False, True]]])

        # Ensure we're normalizing probabilities
        self.assertEqual(result.tolist(), [0.5, 0.5])

    def test_sample(self):
        model = mock_model()

        sequence = model.sample(0)
        self.assertEqual(0, len(sequence))

        sequence = model.sample(100)
        self.assertEqual(100, len(sequence))

        # This assert is non-deterministic, but should always pass.
        # Could in theory get a sequence of all '0's or something though.
        self.assertEqual(set(model.config.model.alphabet), set(sequence))

    def test_translations(self):
        cfg = config()
        cfg['transformations']['translate'] = ["ab", "01"]
        model = mock_model(cfg)
        self.assertEqual(model.transform("abab"), "0101")

    def test_substitutions(self):
        cfg = config()
        cfg['transformations']['substitutions'] = [["ab", "0"], ["ba", "1"]]
        model = mock_model(cfg)
        self.assertEqual(model.transform("abba"), "01")

    def test_translation_and_substitution(self):
        cfg = config()
        cfg['transformations']['translate'] = ["ab", "01"]
        cfg['transformations']['substitutions'] = [["11", "1"], ["01", "1"], ["10", "0"]]
        model = mock_model(cfg)
        self.assertEqual(model.transform("abba"), "0")

    def test_invalid_transformation(self):
        cfg = config()
        cfg['transformations']['translate'] = ["01", "ab"]
        cfg['transformations']['substitutions'] = [["a", "aa"], ["b", "bb"]]
        model = mock_model(cfg)
        with self.assertRaises(Exception):
            model.transform("0101")
