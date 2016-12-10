import unittest
from .lists import rfind, drop_tail_until, take

class TestLists(unittest.TestCase):

    def test_rfind(self):
        self.assertEqual(rfind(None, []), None)
        self.assertEqual(rfind(1, [1, 2, 3, 1, 2, 3]), 3)
        self.assertEqual(rfind(2, [1, 2, 3, 1, 2, 3]), 4)
        self.assertEqual(rfind(3, [1, 2, 3, 1, 2, 3]), 5)
        self.assertEqual(rfind(4, [1, 2, 3, 1, 2, 3]), None)

    def test_drop_tail_until(self):
        self.assertEqual(drop_tail_until(None, []), [])
        self.assertEqual(drop_tail_until(0, [1, 2, 3]), [1, 2, 3])
        self.assertEqual(drop_tail_until(1, [1, 2, 3]), [1])
        self.assertEqual(drop_tail_until(2, [1, 2, 3]), [1, 2])
        self.assertEqual(drop_tail_until(3, [1, 2, 3]), [1, 2, 3])

    def test_take(self):
        self.assertEqual(take(0, []), [])
        self.assertEqual(take(1, []), [])
        self.assertEqual(take(0, [1, 2, 3]), [])
        self.assertEqual(take(1, [1, 2, 3]), [1])
        self.assertEqual(take(2, [1, 2, 3]), [1, 2])
        self.assertEqual(take(3, [1, 2, 3]), [1, 2, 3])
        self.assertEqual(take(4, [1, 2, 3]), [1, 2, 3])
        self.assertEqual(take(-1, [1, 2, 3]), [])
