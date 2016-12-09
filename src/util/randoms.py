from random import SystemRandom
from .packing import MAX_INT

RAND = SystemRandom()

def random_ints():
    """ Generates an infinite stream of 32-bit integers. """
    while True:
        yield RAND.randint(0, MAX_INT)
