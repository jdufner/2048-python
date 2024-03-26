import game.logic as logic
import logging
import unittest


class LogicTestCase(unittest.TestCase):

    def testReverse(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        logging.debug(matrix)
        reversed_matrix = logic.reverse(matrix)
        logging.debug(reversed_matrix)
        self.assertEqual(matrix[0][0], reversed_matrix[0][2])
        self.assertEqual(matrix[0][1], reversed_matrix[0][1])
        self.assertEqual(matrix[0][2], reversed_matrix[0][0])
        self.assertEqual(matrix[1][0], reversed_matrix[1][2])
        self.assertEqual(matrix[1][1], reversed_matrix[1][1])
        self.assertEqual(matrix[1][2], reversed_matrix[1][0])
        self.assertEqual(matrix[2][0], reversed_matrix[2][2])
        self.assertEqual(matrix[2][1], reversed_matrix[2][1])
        self.assertEqual(matrix[2][2], reversed_matrix[2][0])

    def testTranspose(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        logging.debug(matrix)
        transposed_matrix = logic.transpose(matrix)
        logging.debug(transposed_matrix)
        self.assertEqual(matrix[0][0], transposed_matrix[0][0])
        self.assertEqual(matrix[0][1], transposed_matrix[1][0])
        self.assertEqual(matrix[0][2], transposed_matrix[2][0])
        self.assertEqual(matrix[1][0], transposed_matrix[0][1])
        self.assertEqual(matrix[1][1], transposed_matrix[1][1])
        self.assertEqual(matrix[1][2], transposed_matrix[2][1])
        self.assertEqual(matrix[2][0], transposed_matrix[0][2])
        self.assertEqual(matrix[2][1], transposed_matrix[1][2])
        self.assertEqual(matrix[2][2], transposed_matrix[2][2])


if __name__ == '__main__':
    unittest.main()
