import unittest
from random import choice
from encoding import encode, decode
from model import Model

class MockKerasModel(object):
    """ A mock keras model with sequence_length of 5, alphabet of 2 characters,
    and always predicts each character with equal probability.
    """

    def __init__(self):
        self.input_shape = (0, 3, 2)

    def predict(self, sequence, verbose):
        self.last_sequence = sequence
        return [[0.1, 0.1]]

class TestModel(unittest.TestCase):

    def test_init_explicit(self):
        mock = MockKerasModel()
        config = {
            'alphabet': '01',
            'normalizing_length': 10,
            'priming_length': 10,
            'max_padding_trials': 100,
            'padding_novelty_growth_rate': 100,
            'novelty': 2.0,
            'boundary': '1'
        }
        model = Model(mock, config)

        self.assertEqual(model.model, mock)
        self.assertEqual(model.alphabet, sorted(config['alphabet']))
        self.assertEqual(model.normalizing_length, config['normalizing_length'])
        self.assertEqual(model.priming_length, config['priming_length'])
        self.assertEqual(model.max_padding_trials, config['max_padding_trials'])
        self.assertEqual(model.padding_novelty_growth_rate, config['padding_novelty_growth_rate'])
        self.assertEqual(model.boundary, config['boundary'])
        self.assertEqual(model.novelty, config['novelty'])
        self.assertEqual(model.sequence_length, 3)

    def test_init_default(self):
        mock = MockKerasModel()
        config = {
            'alphabet': '01',
            'normalizing_length': 10,
            'priming_length': 10,
            'boundary': '1'
        }
        model = Model(mock, config)

        self.assertEqual(model.model, mock)
        self.assertEqual(model.alphabet, sorted(config['alphabet']))
        self.assertEqual(model.normalizing_length, config['normalizing_length'])
        self.assertEqual(model.priming_length, config['priming_length'])
        self.assertNotEqual(model.max_padding_trials, None)
        self.assertNotEqual(model.padding_novelty_growth_rate, None)
        self.assertNotEqual(model.novelty, None)
        self.assertEqual(model.boundary, config['boundary'])
        self.assertEqual(model.sequence_length, 3)

    def test_validation(self):
        # Alphabet too large
        with self.assertRaises(Exception):
            config = {
                'alphabet': '012',
                'normalizing_length': 0,
                'priming_length': 0,
                'boundary': '1'
            }
            Model(MockKerasModel(), config)

        # Boundary not in alphabet
        with self.assertRaises(Exception):
            config = {
                'alphabet': '012',
                'normalizing_length': 0,
                'priming_length': 0,
                'boundary': '3'
            }
            Model(MockKerasModel(), config)

    def test_predict(self):
        config = {
            'alphabet': '01',
            'normalizing_length': 10,
            'priming_length': 10,
            'boundary': '1'
        }
        model = Model(MockKerasModel(), config)
        sequence = "001"
        result = model.predict(sequence)

        # Ensure we're one-hot encoding
        self.assertEqual(model.model.last_sequence.tolist(), [[[True, False], [True, False], [False, True]]])

        # Ensure we're normalizing probabilities
        self.assertEqual(result.tolist(), [0.5, 0.5])

    def test_translations(self):
        config = {
            'alphabet': '01',
            'normalizing_length': 10,
            'priming_length': 10,
            'boundary': '1',
            'transformations': {'translate': ["ab", "01"]}
        }
        model = Model(MockKerasModel(), config)
        self.assertEqual(model.transform("abab"), "0101")

    def test_substitutions(self):
        config = {
            'alphabet': '01',
            'normalizing_length': 10,
            'priming_length': 10,
            'boundary': '1',
            'transformations': {'substitutions': [["ab", "0"], ["ba", "1"]]}
        }
        model = Model(MockKerasModel(), config)
        self.assertEqual(model.transform("abba"), "01")

    def test_translation_and_substitution(self):
        config = {
            'alphabet': '01',
            'normalizing_length': 10,
            'priming_length': 10,
            'boundary': '1',
            'transformations': {
                'translate': ["ab", "01"],
                'substitutions': [["11", "1"], ["01", "1"], ["10", "0"]]
            }
        }
        model = Model(MockKerasModel(), config)
        self.assertEqual(model.transform("abba"), "0")

    def test_invalid_transformation(self):
        config = {
            'alphabet': '01',
            'normalizing_length': 10,
            'priming_length': 10,
            'boundary': '1',
            'transformations': {
                'translate': ["01", "ab"],
                'substitutions': [["a", "aa"], ["b", "bb"]]
            }
        }
        model = Model(MockKerasModel(), config)
        with self.assertRaises(Exception):
            model.transform("0101")
