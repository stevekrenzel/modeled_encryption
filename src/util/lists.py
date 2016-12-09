from itertools import chain

def rfind(x, xs):
    """Finds the right-most index of an element in a list.

    Example:
        >> rfind(3, [1, 2, 3, 1, 2, 3])
        5
        >> rfind(4, [1, 2, 3, 1, 2, 3])
        None

    Args:
        x (obj): The element to find the index of.

        xs (list): The list to search through.

    Returns:
        The index of the right-most occurence of `x` in `xs`, or None if it
        isn't in the list.
    """
    for i in range(len(xs) - 1, -1, -1):
        if xs[i] == x:
            return i
    return None

def trim_tail(x, xs):
    """Similar to 'trim' on a string, this will chop off consecutive occurences
    of `x` that are present at the end of `xs`.

    Example:
        >> trim_tail(' ', ['a', 'b', 'c', ' ', ' ', ' '])
        ['a', 'b', 'c']
        >> trim_tail("!", "hello!!")
        'hello'
        >> trim_tail(1, [1, 1, 1, 2]
        [1, 1, 1, 2]

    Args:
        x (obj): The value to trim from the end of `xs`.

        xs (list): The list of items to trim.

    Returns:
        A new list that is equivalent to `xs` with the trailing `x` items
        removed.
    """
    cutoff = len(xs)
    for i in range(len(xs) - 1, -1, -1):
        if xs[i] != x:
            break
        cutoff = i
    return xs[:cutoff]

def drop_tail_until(x, xs):
    """Removes all right-most elements from a list until a value is found.

    Example:
        >> drop_tail_until(1, [1, 2, 3, 1, 2, 3])
        [1, 2, 3, 1]
        >> drop_tail_until(4, [1, 2, 3, 1, 2, 3])
        [1, 2, 3, 1, 2, 3]

    Args:
        x (obj): The stopping value for dropping elements.

        xs (list): The list to drop values from.

    Returns:
        A copy of `xs` with anything after the right-most `x` value removed.
    """
    last = rfind(x, xs)
    if last == None:
        return xs[:]
    return xs[:last + 1]

def take(n, xs):
    """Takes the first `n` elements of `xs`.

    Example:
        >> gen = (i for i in range(5))
        >> take(3, gen)
        [0, 1, 2]
        >> take(2, gen)
        [3, 4]
        >> take(1, gen)
        []

    Args:
        n (int): The number of items to take from `xs`.

        xs (sequence): The sequence to take items from.

    Returns:
        A list of the first `n` items in `xs`. If `xs` doesn't have
        enough items then as many as can be drawn will be returned.
    """
    return [x for (_, x) in zip(range(n), xs)]

def to_generator(xs):
    """ Converts a sequence to a generator. """
    return (x for x in xs)
