import unittest
import numpy as np

class AgentTestCase(unittest.TestCase):

    def testMatrixToArray(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        #print(matrix)
        array = np.asanyarray(matrix).ravel()
        #print(array)
        self.assertEqual(1, array[0])
        self.assertEqual(9, array[8])

    def testMatrixEqual(self):
        matrixA = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        matrixB = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(matrixA, matrixB)

    def testMatrixNotEqual(self):
        matrixA = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        matrixB = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        self.assertNotEqual(matrixA, matrixB)

if __name__ == '__main__':
    unittest.main()
