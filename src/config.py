""" Handles loading / parsing config files. """
# XXX - This impl stretches namedtuples a little far, doesn't do type
# validation on values, and gives a shitty / not very helpful error message
# when a field is missing. This can all be improved.

import json
from os.path import join, normpath, dirname
from collections import namedtuple

ModelConfig = namedtuple('ModelConfig', [
# Configuration values related to the composition of the model.

    'alphabet',
    # A string containing the entire alphabet of the model.

    'nodes',
    # The number of nodes in the hidden layer of the the LSTM.

    'sequence_length',
    # The length of the input sequence to the model.

    'boundary',
    # The delimiting character between tokens in the model. (e.g. ' ' for a
    # model that generates words, or '.' for a model that generates sentences)

    'weights_file',
    # The path to the file containing the model's weights. When training, the
    # weights will be stored here.
])

EncodingConfig = namedtuple('EncodingConfig', [
# Configuration values for encoding sequences.

    'normalizing_length',
    # The number of characters to run through a randomized model to normalize
    # it's output distribution.

    'priming_length',
    # The number of characters to generate to create an initial sequence for
    # encoding.

    'max_padding_trials',
    # The maximum number of times to try adding padding before giving up.

    'padding_novelty_growth_rate',
    # A float representing the increase in novelty after each failed attempt at
    # padding. We increase the novelty slightly after each failure in an attempt
    # to increase the odds of generating padding of a valid length while
    # minimizing the skew to the model's output distribution.

    'novelty',
    # The novelty (a.k.a. temperature) to use when normalizing prediction
    # weights. A smaller number will make the predictions more conservative.
])

TrainingConfig = namedtuple('TrainingConfig', [
# Configuration values for training the model.

    'validation_split',
    # The percentage of data to be withheld for validation suring training.

    'batch_size',
    # The size of batches during training.

    'epochs',
    # The number of epochs (complete passes of data) to train on.
])

TransformationsConfig = namedtuple('TransformationsConfig', [
# Configuration values describing transformations for data input to the model.
# Note: Translations are ran before substitutions.

    'translate',
    # Optional
    # A tuple containg two strings of same length. Characters from the first
    # string will be replaced with corresponding characters from the second string.

    'substitutions',
    # Optional
    # A list of regular expressions and the strings that should replace their matches.
])

ConfigConstructor = namedtuple('Config', [
# Container for configuration values.

    'model',
    # ModelConfig

    'encoding',
    # EncodingConfig

    'training',
    # TrainingConfig

    'transformations'
    # TransformationsConfig
])

class ValidationError(Exception):
    """ Exception thrown on validation issues. """
    pass

def build_namedtuple(constructor, values, optional):
    """ Given a namedtuple constructor and a dictionary, this ensures that
    the dictionary has the fields required for the namedtuple and then
    constructs it.

    If `optional` is True, then it defaults to a value of None for non-existent
    keys.

    If a key is missing and `optional` is not True, then a validation error is
    raised.

    Example:
        >> Foo = namedtuple('Foo', ['x', 'y'])
        >> values = {'x': 1, 'y': 2, 'z': 3}
        >> build_namedtuple(Foo, values)
        Foo(x=1, y=2)

        >> values = {'x': 1}
        >> build_namedtuple(Foo, values, optional=True)
        Foo(x=1, y=None)

    Args:
        constructor (namedtuple constructor): The namedtuple to build.

        values (dict): The key/value pairs to instantiate the namedtuple with.

        optional (bool): Whether or not to require all fields from the
        namedtuple to be present in `values`. If True, then missing fields
        default to None.

    Returns:
        Instantiated tuple.

    Raises:
        ValidationError: On missing fields when `optional` is False.
    """
    for field in constructor._fields:
        if not optional and field not in values:
            raise ValidationError("Field '%s' is  missing from '%s'" % (field, constructor.__name__))
    return constructor(**{k:values.get(k, None) for k in constructor._fields})

def Config(kv):
    """ Given a dictionary of key/values, creates a Config object from those
    values.

    Args:
        kv (dict) - The values to use for creating the config.

    Returns:
        An instantiated Config object.

    Raises:
        ValidationError: If a required field is missing.

        ValidationError: If an invalid boundary character is provided.
    """
    constructors = [('model'          , ModelConfig          , False),
                    ('encoding'       , EncodingConfig       , False),
                    ('training'       , TrainingConfig       , False),
                    ('transformations', TransformationsConfig, True )]

    tuples = {key: build_namedtuple(constructor, kv[key], optional)
              for (key, constructor, optional) in constructors if key in kv}

    config = build_namedtuple(ConfigConstructor, tuples, optional=False)

    if config.model.boundary not in config.model.alphabet:
        raise ValidationError("The boundary must be a character present in the alphabet.")

    return config

def load_config(config_file):
    """ Reads a json file and constructs a Config object from it.

    Args:
        config_file (string): The path to the json file to parse.

    Returns:
        An instantiated Config object.

    Raises:
        Error: If `config_file` doesn't exist.

        ValidationError: If required keys are missing.

        ValidationError: If other config validation fails.
    """

    with open(config_file) as config_handle:
        raw = json.load(config_handle)

    if 'model' in raw and 'weights_file' in raw['model']:
        weights_file = raw['model']['weights_file']
        directory = dirname(config_file)
        normalized = normpath(join(directory, weights_file))
        raw['model']['weights_file'] = normalized

    return Config(raw)
