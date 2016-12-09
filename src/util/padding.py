from random import SystemRandom
from .modeling import recite
from .lists import drop_tail_until, trim_tail
from .packing import BYTES_IN_INT
from .randoms import random_ints

RAND = SystemRandom()

def pad(model, initial, values, blocksize):
    """Extends the provided values with model predictions to make the total
    length of (initial + values) equal to a multiple of blocksize.

    We do this by continuously generating potential suffixes until a boundary
    is hit. Once a boundary is hit (e.g. a full token is formed), we split the
    token into prefixes that when appended to the provided payload will result
    in a new payload that is the correct length.

    Of these candidate prefixes, we then uniformly sample them and choose one
    to use as padding.

    This process ensures that the payload always ends with a partial or
    complete token (with or without the trailing boundary character).

    Example:
        >> initial = list("THIS IS AN INITIAL SEQUENCE FOR AN EXAMPLE FOOBAR ")
        >> padd(model, initial, list("HELLO"), 16)
        ['H', 'E', 'L', 'L', 'O', ' ', 'M', 'U', 'C']
        >> pad(model, initial, list("FOO "), 16)
        ['F', 'O', 'O', ' ', 'F', 'R', 'O', 'M', ' ']
        >> "".join(_)
        'FOO FROM '

    Args:
        model (Model): The model to use for encoding.

        initial (list): The initial sequence used to seed the model.

        values (list): The values to pad.

        blocksize (int): The mutliple that we need to pad to.

    Returns:
        A list of values + padding.

    Raises:
        Exception: If padding fails to generate. This is a non-deterministic
        process, so trying again may work. This is extremely unlikely to be
        raised if you have any reasonable model.padding_novelty_growth_rate and
        model.max_padding_trials.
    """
    if values[-1] != model.boundary:
        values = values + [model.boundary]

    length = _base_length(model, values)
    block_capacity = blocksize // BYTES_IN_INT
    first_length = block_capacity - (length % block_capacity)
    joined = initial + values

    for token in _tokens(model, joined):
        if len(token) >= first_length:
            offsets = range(first_length, len(token) + 1, block_capacity)
            token_prefixes = [token[:j] for j in offsets]
            padding = RAND.choice(token_prefixes)
            return values + padding

    raise Exception("Failed to generate padding. This is non-deterministic. Run again or try increasing padding_novelty_growth_rate count.")

def unpad(model, values):
    """Removes the last token (including any trailing boundaries) from values.

    Example:
        >> unpad(model, "FOO BAR ")
        'FOO '
        >> unpad(model, "HELLO MUC")
        'HELLO '
        >> unpad(model, "HELLO ")
        'HELLO'
        >> unpad(model, "HELLO")
        'HELLO'

    Args:
        model (Model): The model that was used to pad this input.

        values (list): The values to remove padding from.

    Returns:
        A copy of `values` with the last token removed, or unchanged if there
        is only one token.
    """
    trimmed = trim_tail(model.boundary, values)
    return drop_tail_until(model.boundary, trimmed)

def _tokens(model, base):
    """ Generates a stream of tokens with increasing novelty. """
    for novelty in _novelities(model):
        yield _generate_token(model, base, novelty)

def _novelities(model):
    """ A sequence of increasing novelities. """
    for i in range(model.max_padding_trials):
        yield model.novelty * (model.padding_novelty_growth_rate ** i)

def _generate_token(model, start, novelty):
    """Generates a random single token following the `start` sequence.

    Note: If the probability of generating a boundary character is low then
    this could take a while to run. It won't stop trying to generate values
    until it succeeds in generating a boundary.

    Example:
        >> initial = list("THIS IS AN INITIAL SEQUENCE FOR AN EXAMPLE FOOBAR ")
        >> _generate_token(model, initial, 1.0)
        ['A', 'N', 'D', ' ']
        >> _generate_token(model, initial, 1.0)
        ['M', 'E', 'A', 'N', 'S', ' ']

    Args:
        model (Model): The model to be used for prediction.

        start (list): The sequence to generate a subsequent token for.

        novelty (float): The conservativeness of the token generation.

    Returns:
        A random token that the model believes would reasonably follow the
        provided sequence.
    """
    # Non-deterministic. If `boundary` has a low prob of being generated, this
    # could take a while to run.
    stream = recite(model, start, random_ints(), novelty) # Infinite stream
    token = []
    for c in stream:
        token.append(c)
        if c == model.boundary:
            return token

def _base_length(model, values):
    """ Returns the length of the payload without padding. """
    init = model.sequence_length
    norm = model.normalizing_length
    prim = model.priming_length
    return init + norm + prim + len(values)
