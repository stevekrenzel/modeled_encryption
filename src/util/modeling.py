from .sampling import choose_choice, choose_weight
from .math import scale
from .packing import MAX_INT

def tabulate(model, initial, values, novelty=None):
    """Given a sequence of values, this returns a list of random integer
    weights drawn from to ranges corresponding to the model's probability of
    predicting each value.

    Example:
        >> initial = list("THIS IS AN INITIAL SEQUENCE FOR AN EXAMPLE FOOBAR ")
        >> list(tabulate(model, initial, "HELLO"))
        [3248025205, 3874735365, 4292362767, 3915527017, 4267391621]
        >> "".join(recite(model, initial, _))
        "HELLO"

    Args:
        model (Model): The model to use for predictions.

        initial (list): The initial sequence to feed to the model.

        values (list): The values to generate weights for.

        novelty (float, optional): The conservativeness of the predictions.

    Return (generator(int)):
        A list of randomized weights corresponding to the model's probability
        of predicting each value.
    """
    alphabet = model.config.model.alphabet

    def fn(value, weights):
        weight = choose_weight(value, alphabet, weights)
        return (value, weight)

    return _scan_model(model, fn, initial, values, novelty)

def recite(model, initial, weights, novelty=None):
    """Given a sequence of weights, this returns the values that correspond to
    the model's predictions for each weight.

    Example:
        >> initial = list("THIS IS AN INITIAL SEQUENCE FOR AN EXAMPLE FOOBAR ")
        >> weights = [3248025205, 3874735365, 4292362767, 3915527017, 4267391621]
        >> list(recite(model, initial, weights))
        ['H', 'E', 'L', 'L', 'O']

    Args:
        model (Model): The model to use for predictions.

        initial (list): The initial sequence to feed to the model.

        weights (list(int)): The weights to use when choosing values.

        novelty (float, optional): The conservativeness of the predictions.

    Returns (generator):
        A sequence of values as chosen by the provided weights.
    """
    alphabet = model.config.model.alphabet

    def fn(weight, weights):
        value = choose_choice(weight, alphabet, weights)
        return (value, value)

    return _scan_model(model, fn, initial, weights, novelty)

def _scan_model(model, fn, init, xs, novelty=None):
    """For every value in `xs`, this calls `fn` with both the value and the
    weights of the model's current predictions. The sequence being fed to the
    model is then updated, and we repeat the process.

    The function, `fn`, returns two values. This function accumulates the
    second of these values and returns them in a generator.

    This function is similar in spirit to Haskell's scanl, but applied to a
    model instead of a list. Scanl is just like reduce in Python, but every
    intermediate value gets returned instead of just the final accumulated
    value.

    See `tabulate` and `recite` for example usage.

    Args:
        model (Model): The model to use for predictions.

        fn (function): A function that takes a value and a list of weights and
            returns the next value in a sequence for the model, as well as a
            value to be accumulated and returned from this function.

        init (list): The initial values to feed to the model.

        xs (list): The values to feed to `fn`.

        novelty (float): The conservativeness of the predictions.

    Returns (generator):
        A sequence of the values computed by `fn`.
    """
    sequence_length = model.config.model.sequence_length
    sequence = init[-sequence_length:]
    for x in xs:
        probabilities = model.predict(sequence, novelty)
        # We use (MAX_INT + 1) because weights are chosen 0 <= w <= MAX_INT
        scaled = scale(probabilities, MAX_INT + 1, lowest=1)
        (next_value, y) = fn(x, scaled)
        yield y
        sequence = (sequence + [next_value])[-sequence_length:]
