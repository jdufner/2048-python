import logging
import torch
from torch import Tensor
import unittest


class TorchTestCase(unittest.TestCase):

    def testCudaIsAvailable(self):
        is_cuda_available: bool = torch.cuda.is_available()
        logging.debug(f'CUDA is available: {is_cuda_available}')
        self.assertTrue(is_cuda_available or not is_cuda_available)

    def testCudaDevice(self):
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.debug(f'CUDA device: {device.type}')
        self.assertTrue(True)

    def testRandomTensor(self):
        matrix: Tensor = torch.rand(5, 3)
        logging.debug(matrix)
        self.assertEqual(3, matrix.size(dim=1))


if __name__ == '__main__':
    unittest.main()
