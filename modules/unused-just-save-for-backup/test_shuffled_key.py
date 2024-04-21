import os
import torch
import unittest

import sys
sys.path.append("..") 
from hextree import key2txyz, txyz2key


class ShuffledKeyTest(unittest.TestCase):

    def test_shuffled_key_rand_case(self):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for d in devices:
            t = torch.randint(16384, (1000,), device=d)
            x = torch.randint(16384, (1000,), device=d)
            y = torch.randint(16384, (1000,), device=d)
            z = torch.randint(16384, (1000,), device=d)
            b = torch.randint(128, (1000,), device=d)

            key = txyz2key(t, x, y, z, b, 14)
            t1, x1, y1, z1, b1 = key2txyz(key, 14)

            self.assertTrue((t1 == t).all())
            self.assertTrue((x1 == x).all())
            self.assertTrue((y1 == y).all())
            self.assertTrue((z1 == z).all())
            self.assertTrue((b1 == b).all())
        
    def test_shuffled_key_spec_case(self):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for d in devices:
            t = torch.tensor([0, 1, 1, 2, 3, 3, 4, 8], device=d)
            x = torch.tensor([0, 3, 5, 6, 2, 3, 4, 5], device=d)
            y = torch.tensor([4, 23, 45, 67, 6, 6, 76, 54], device=d)
            z = torch.tensor([12, 4, 5, 6, 56, 90, 10, 23], device=d)
            b = torch.tensor([0, 0, 1, 2, 3, 4, 4, 4], device=d)

            key = txyz2key(t, x, y, z, b, 7)
            key_gt = torch.tensor([
                0b0000000000000000000000000000000000000000000000000001001100000000,
                0b0000000000000000000000000000000000000000000000100000001101101110,
                0b0000000100000000000000000000000000000000001000000010011100001111,
                0b0000001000000000000000000000000000000010000000000000010111110010,
                0b0000001100000000000000000000000000000000000100010001001011101000,
                0b0000010000000000000000000000000000000001000000010001001011111100,
                0b0000010000000000000000000000000000000010000000000011111000010000,
                0b0000010000000000000000000000000000000000001000111000011100110101
            ], device=d)

            self.assertTrue((key == key_gt).all())


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    unittest.main()