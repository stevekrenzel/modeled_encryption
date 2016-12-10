import unittest
from .packing import pack_ints, unpack_ints

class TestPacking(unittest.TestCase):

    def test_packing(self):
        self.assertEqual(unpack_ints(pack_ints([])), ())
        self.assertEqual(unpack_ints(pack_ints([1])), (1,))
        self.assertEqual(unpack_ints(pack_ints([1, 2])), (1, 2))
        self.assertEqual(unpack_ints(pack_ints([1, 2, 3])), (1, 2, 3))
