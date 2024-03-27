import numpy as np
import unittest


class AgentTestCase(unittest.TestCase):

    def testMatrixToArray(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        #print(matrix)
        array = np.asanyarray(matrix).ravel()
        #print(array)
        self.assertEqual(1, array[0])
        self.assertEqual(9, array[8])

    def testMatrixEqual(self):
        matrix_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        matrix_b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(matrix_a, matrix_b)

    def testMatrixNotEqual(self):
        matrix_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        matrix_b = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        self.assertNotEqual(matrix_a, matrix_b)


if __name__ == '__main__':
    unittest.main()
