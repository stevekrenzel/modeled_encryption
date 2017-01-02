from config import Config
from model import Model

class MockKerasModel(object):
    """ A mock keras model with sequence_length of 5, alphabet of 2 characters,
    and always predicts each character with equal probability.
    """
    def __init__(self, base):
        sequence_length = base.config.model.sequence_length
        self.alphabet_size = len(base.config.model.alphabet)
        self.input_shape = (0, sequence_length, self.alphabet_size)

    def predict(self, sequence, verbose):
        self.last_sequence = sequence
        return [[1/self.alphabet_size] * self.alphabet_size]

class MockModel(Model):
    def _create_model(self):
        return MockKerasModel(self)

def config():
    """ Returns a copy of the config. """
    return {
        'model': {
            'alphabet': '012',
            'nodes': 0,
            'sequence_length': 0,
            'boundary': '0',
            'weights_file': '/dev/null',
        },
        'encoding': {
            'normalizing_length': 0,
            'priming_length': 0,
            'max_padding_trials': 1000,
            'padding_novelty_growth_rate': 1.01,
            'novelty': 0.5,
        },
        'training': {
            'validation_split': 0.05,
            'batch_size': 32,
            'epochs': 100,
        },
        'transformations': {
        },
    }

DEFAULT_CONFIG = config()

def mock_model(config=DEFAULT_CONFIG):
    return MockModel(Config(config))
