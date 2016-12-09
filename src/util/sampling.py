from random import SystemRandom

RAND = SystemRandom()

def choose_choice(weight, choices, weights):
    """This is just like a normal random weighted sample, but we provide the
    chosen weight instead of it being randomly generated.

    Given a list of items and associated weights, this returns the item
    associated with the provided `weight`.

    The weights are integer weights. Associating a weight of `N` with an item
    is equivalent to repeating it's corresponding item `N` times in a uniform
    sample. That is, given choices ["A", "B"] and weights [2, 3], this is
    equivalent to uniformly sampling ["A", "A", "B", "B", "B"] (but more memory
    efficient).

    Examples:
        >> choose_choice(0, ["A", "B", "C"], [1, 2, 3])
        "A"

        >> choose_choice(1, ["A", "B", "C"], [1, 2, 3])
        "B"

        >> choose_choice(2, ["A", "B", "C"], [1, 2, 3])
        "B"

        >> choose_choice(3, ["A", "B", "C"], [1, 2, 3])
        "C"

        >> choose_choice(4, ["A", "B", "C"], [1, 2, 3])
        "C"

        >> choose_choice(5, ["A", "B", "C"], [1, 2, 3])
        "C"

    Args:
        weight (int): A number >= 0 and < sum(weights). This is used
            to determine which element in the `choices` list to select.

        choices (list(x)): The list of items to choose from.

        weights (list(int)): The integer weights associated with each
            item in `choices`.

    Returns:
        The choice corresponding to the provided weight.

    Raises:
        ValueError: If `choices` and `weights` have different lengths.

        ValueError: If `weight` is less than zero or greater than sum(weights).
    """
    if len(weights) != len(choices):
        raise ValueError("Weights has length %s, but choices has length %s."
                         % (len(weights), len(choices)))

    if weight < 0:
        raise ValueError("Weight, %s, can not be less than zero." % (weight))

    total = 0
    for c, w in zip(choices, weights):
        total = total + w
        if weight < total:
            return c

    raise ValueError("Weight, %s, is larger than %s, the sum of all weights."
                    % (weight, total))

def choose_weight(choice, choices, weights):
    """This is the opposite of choose_choice. Given a choice, this generates a
    random weight corresponding to the provided choice. That is, choose_choice
    returns a choice given a weight, whereas choose_weight returns a weight for
    a provided choice.

    Each item in `choices` has a range of weights that corresponds to it. Given
    a specific `choice`, this returns a random weight from within that range.

    More explicitly, given:

        choice: "B"
        choices: ["A", "B"]
        weights: [2, 3]

    The corresponding weights for each choice are:

        choices: A, A, B, B, B
        weights: 0, 1, 2, 3, 4

    The end result is calling `choose_choice` with the weight returned from
    this function will always result in the `choice` passed into this function:

        >> weight = choose_weight(choice, choices, weights)
        >> sample = choose_choice(weight, choices, weights)
        >> sample == choice
        True

    Examples:
        This is deterministic because each choice has one valid weight:

        >> sample_weights("A", ["A", "B", "C"], [1,1,1])
        0

        This is non-deterministic:

        >> sample_weights("C", ["A", "B", "C"], [1,2,3])
        4

    Args:
        choice (x): The item to generate a weight for.

        choices (list(x)): The items being sampled from.

        weights (list(int)): The weights of each item in `choices`.

    Returns:
        A random weight from the interval corresponding to the given `choice`.
        Returns None if the choice has a weight of zero.

    Raises:
        ValueError: If the length of `choices` is not the same as the length of
                    `weights`.
    """
    if len(weights) != len(choices):
        raise ValueError("Weights has length %s, but choices has length %s."
                         % (len(weights), len(choices)))

    start, end = 0, 0
    for c, w in zip(choices, weights):
        start, end = end, end + w
        if c == choice:
            break

    if start == end:
        return None # When weight is zero

    return RAND.randint(start, end - 1)
