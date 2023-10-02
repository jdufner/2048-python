import unittest
import logic

class LogicTestCase(unittest.TestCase):

    def testReverse(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        #print(matrix)
        reversedMatrix = logic.reverse(matrix)
        #print(reversedMatrix)
        self.assertEqual(matrix[0][0], reversedMatrix[0][2])
        self.assertEqual(matrix[0][1], reversedMatrix[0][1])
        self.assertEqual(matrix[0][2], reversedMatrix[0][0])
        self.assertEqual(matrix[1][0], reversedMatrix[1][2])
        self.assertEqual(matrix[1][1], reversedMatrix[1][1])
        self.assertEqual(matrix[1][2], reversedMatrix[1][0])
        self.assertEqual(matrix[2][0], reversedMatrix[2][2])
        self.assertEqual(matrix[2][1], reversedMatrix[2][1])
        self.assertEqual(matrix[2][2], reversedMatrix[2][0])

    def testTranspose(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        #print(matrix)
        transposedMatrix = logic.transpose(matrix)
        #print(transposedMatrix)
        self.assertEqual(matrix[0][0], transposedMatrix[0][0])
        self.assertEqual(matrix[0][1], transposedMatrix[1][0])
        self.assertEqual(matrix[0][2], transposedMatrix[2][0])
        self.assertEqual(matrix[1][0], transposedMatrix[0][1])
        self.assertEqual(matrix[1][1], transposedMatrix[1][1])
        self.assertEqual(matrix[1][2], transposedMatrix[2][1])
        self.assertEqual(matrix[2][0], transposedMatrix[0][2])
        self.assertEqual(matrix[2][1], transposedMatrix[1][2])
        self.assertEqual(matrix[2][2], transposedMatrix[2][2])

if __name__ == '__main__':
    unittest.main()
