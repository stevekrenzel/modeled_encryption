from model import Model

class MockKerasModel(object):
    """ A mock keras model with sequence_length of 5, alphabet of 2 characters,
    and always predicts each character with equal probability.
    """
    def __init__(self, base):
        self.alphabet_size = len(base.alphabet)
        self.input_shape = (0, base.sequence_length, self.alphabet_size)

    def predict(self, sequence, verbose):
        self.last_sequence = sequence
        return [[1/self.alphabet_size] * self.alphabet_size]

config = {
    'alphabet': '012',
    'nodes': 0,
    'sequence_length': 0,
    'normalizing_length': 0,
    'priming_length': 0,
    'max_padding_trials': 1000,
    'boundary': '0'
}

def mock_keras(config=config):
    return Model(config, lambda base: MockKerasModel(base))
