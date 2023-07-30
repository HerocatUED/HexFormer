# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# Modified by Xiang Wang
# --------------------------------------------------------

import os
import torch
import unittest

import sys 
sys.path.append('..')
from hextree.shuffled_key import key2txyz, txyz2key


class ShuffledKeyTest(unittest.TestCase):

    def test_shuffled_key(self):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        for d in devices:
            t = torch.randint(1024, (10000,), device=d)
            x = torch.randint(8192, (10000,), device=d)
            y = torch.randint(8192, (10000,), device=d)
            z = torch.randint(8192, (10000,), device=d)
            b = torch.randint(128, (10000,), device=d)

            key = txyz2key(t, x, y, z, b, depth=14)
            t1, x1, y1, z1, b1 = key2txyz(key, depth=14)

            self.assertTrue((t1 == t).all() & (x1 == x).all() & (y1 == y).all() &
                            (z1 == z).all() & (b1 == b).all())


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    unittest.main()
