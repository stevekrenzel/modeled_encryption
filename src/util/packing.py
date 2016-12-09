from struct import pack, unpack, calcsize

BITS_IN_BYTE = 8
BYTES_IN_INT = calcsize('I')
INT_SIZE = BYTES_IN_INT * BITS_IN_BYTE
MAX_INT = (2**INT_SIZE) - 1

def pack_ints(xs):
    """Serializes a list of 32-bit integers into a byte string.

    Example:
        >> pack_ints([1, 2, 3])
        b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00'

    Args:
        xs (list(int)): The list of integers to encode.

    Returns:
        A byte string encoding the list of integers.

    Raises:
        Error: If `xs` does not contain integers.

        Error: If a value is not 0 <= x < 2^32.
    """
    return pack('I' * len(xs), *xs)

def unpack_ints(data):
    """Deserializes a byte string into 32-bit integers.

    Example:
        >> unpack_ints(pack_ints([1,2,3]))
        (1, 2, 3)

    Args:
        data (bytes): The bytes to deserialize.

    Returns:
        A tuple of the decoded integers.

    Raises:
        Error: If the data length is not a multiple of 4.
    """
    return unpack('I' * (len(data) // BYTES_IN_INT), data)
