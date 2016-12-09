import numpy as np

def log_normalize(values, temperature):
    """Normalizes a list of values such that they sum to 1.0, while also
    skewing the distribution based off of `temperature` to amplify or suppress
    the impact of larger values.

    Large `temperature`s will cause the resultant values to be more
    uniform. Small `temperature`s will amplify the distances between values.

    Example:
        >> log_normalize([1, 1], 1.0)
        [0.5, 0.5]
        >> log_normalize([1, 2, 3], 1.0)
        [0.17, 0.33, 0.5]
        >> log_normalize([1, 2, 3], 3.0)
        [0.27,  0.34,  0.39]
        >> log_normalize([1, 2, 3], 0.25)
        [0.01, 0.16, 0.83]

    Args:
        values (list(num)): The list of values to normalize.

        temperature (float): The amount to skew the resultant distribution.

    Returns (numpy.array):
        A normalizes list of values.
    """
    preds = np.asarray(values).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return preds.ravel()

def scale(values, total, lowest=0):
    """Takes a list of values that sum to 1.0 and scales them to integer values
    that sum to `total`, while maintaining their relative ratios. Due to
    rounding, the ratios may change slightly. The impact of this can be
    minimized by choosing a large value for `total`.

    The `lowest` argument puts a floor on the smallest element that will be in
    the resultant list. If this value is non-zero, it ensures that small values
    will still have weight in the resultant list.

    Example:
        >> scale([0.5, 0.5], 10)
        [5, 5]
        >> scale([0.0, 0.5], 10)
        [0, 10]
        >> scale([0.0, 0.5], 10, 1)
        [1, 9]
        >> scale([0.05, 0.2, 0.75], 100)
        [5, 20, 75]
        >> scale([0.005, 0.245, 0.75], 100) # Without `lowest` set, 0.005 gets zero weight in the result
        [0, 24, 76]
        >> scale([0.005, 0.245, 0.75], 100, 1) # With `lowest` set, 0.005 now gets some weight
        [1, 24, 75]

    Args:
        values (list(float)): The list of values to scale.

        total (int): The value that the scaled list will sum to.

        lowest (int, optional): The lowest value allowed in the resultant list.

    Returns (list(int)):
        A list that is scaled to sum to `total`.
    """
    scaled = [int(max(lowest, round(p * total))) for p in values]

    delta = total - sum(scaled)
    max_index = max(range(len(scaled)), key=lambda i: scaled[i])
    scaled[max_index] += delta

    return scaled
