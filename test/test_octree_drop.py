# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# Hextree version modified by Xiang Wang
# --------------------------------------------------------

import os
import torch
import unittest

from .utils import get_hextree
from ..modules.hextree_drop import HextreeDropPath


class HextreeDropTest(unittest.TestCase):

    def test_hextree_drop_path(self):
        r'''Just execute the `OctreeDropPath`, and there are no comparisons with 
        ground-truth results.
        '''

        hextrees = [get_hextree(i) for i in ([4] * 8 + [5] * 2)]
        hextree = hextree.merge_hextrees(hextrees)

        # Test 1
        depth = 5
        nnum = hextree.nnum[depth]
        data = torch.rand(nnum, 4)  # TODO 3 or 4
        drop_path = HextreeDropPath(drop_prob=0.8, nempty=False)
        output = drop_path(data, hextree, depth)

        # Test 2
        nnum_nempty = hextree.nnum_nempty[depth]
        data = torch.rand(nnum_nempty, 4)  # TODO 3 or 4
        drop_path = HextreeDropPath(drop_prob=0.8, nempty=True)
        output = drop_path(data, hextree, depth)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    unittest.main()
