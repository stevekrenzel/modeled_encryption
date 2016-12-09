from numpy import zeros, argmax, array
import numpy as np

def one_hot_encoding(xs, classes):
    """Given a list of values, converts them to one-hot encoding.

    One-hot encoding is an encoding where if you have N distinct possible
    classes, the value gets encoded as an array of size N where all values are
    zero except for the index corresponding to the class of the current value.

    Example:
        >> one_hot_encoding([0, 1, 2], range(3))
        [[ True, False, False],
         [False,  True, False],
         [False, False,  True]]

        >> one_hot_encoding("ace", "abcde")
        [[ True, False, False, False, False],
         [False, False,  True, False, False],
         [False, False, False, False,  True]]

    Args:
        xs (list): The list of values to encode.

        classes (sequence): A sequence containing one of each possible class.

    Returns (numpy.array(bool)):
        A two-dimensional numpy array where each row contains an encoded value.

    Raises:
        ValueError: If a value is present in `xs` that isn't in `classes`.
    """
    # Maps a class to it's corresponding index
    lookup = dict((x, i) for i, x in enumerate(classes))

    X = zeros((len(xs), len(classes)), dtype=np.bool)
    for i, x in enumerate(xs):
        if x not in lookup:
            raise ValueError("Value '%s' is not present in `classes`" % (x))

        X[i, lookup[x]] = True

    return X
