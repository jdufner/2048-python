import unittest
import numpy as np


class PuzzleTestCase(unittest.TestCase):

    def testMaxIndex(self):
        array = [0, 0, 1]
        self.assertEqual(2, np.argmax(array))

    def testMax(self):
        matrix: list = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(6, np.max(matrix))


if __name__ == '__main__':
    unittest.main()
