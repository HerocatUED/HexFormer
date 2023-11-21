import os
import torch
import unittest
import sys

from utils import get_hextree
sys.path.append('..')
from modules import HextreeDropPath


class HextreeDropTest(unittest.TestCase):

    def test_hextree_drop_path(self):
        r'''Just execute the `OctreeDropPath`, and there are no comparisons with 
        ground-truth results.
        '''

        hextrees = [get_hextree(i) for i in (0,1)]
        # htree = hextree.merge_hextrees(hextrees)
        htree = hextrees[0]

        # Test 1
        depth = 5
        nnum = htree.nnum[depth]
        data = torch.rand(nnum, 4)  # TODO 3 or 4
        drop_path = HextreeDropPath(drop_prob=0.8, nempty=False)
        output = drop_path(data, htree, depth)

        # Test 2
        nnum_nempty = htree.nnum_nempty[depth]
        data = torch.rand(nnum_nempty, 4)  # TODO 3 or 4
        drop_path = HextreeDropPath(drop_prob=0.8, nempty=True)
        output = drop_path(data, htree, depth)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    unittest.main()
