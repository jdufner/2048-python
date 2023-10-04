import unittest
import numpy as np
#from agent import Agent

class AgentTestCase(unittest.TestCase):

    def testReverse(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        #print(matrix)
        array = np.asanyarray(matrix).ravel()
        #print(array)
        self.assertEqual(1, array[0])
        self.assertEqual(9, array[8])

if __name__ == '__main__':
    unittest.main()
