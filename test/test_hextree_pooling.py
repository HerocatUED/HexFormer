import os
import torch
import unittest
import sys

from utils import get_hextree
sys.path.append('..')
from modules import HextreeAvgPool, HextreeMaxPool, HextreeMaxUnpool


class HextreePoolTest(unittest.TestCase):
    # only dimension check, no value check
    def test_hextree_max_pooling(self):
        htree = get_hextree(0)
        depth = 5

        nnum_nempty = htree.nnum_nempty[depth]
        data = torch.rand(nnum_nempty, 100)
        max_pooling = HextreeMaxPool(nempty=True, return_indices=True)
        out, indices = max_pooling(data, htree, depth)
        assert out.shape[0] == htree.nnum_nempty[depth-1]

        max_unpooling = HextreeMaxUnpool(nempty=True)
        out = max_unpooling.forward(out, indices, htree, depth-1)
        assert out.shape[0] == htree.nnum_nempty[depth]

        nnum = htree.nnum[depth]
        data = torch.rand(nnum, 100)
        max_pooling = HextreeMaxPool(nempty=False, return_indices=True)
        out, indices = max_pooling(data, htree, depth)
        assert out.shape[0] == htree.nnum[depth-1]

        max_unpooling = HextreeMaxUnpool(nempty=False)
        out = max_unpooling.forward(out, indices, htree, depth-1)
        assert out.shape[0] == htree.nnum[depth]
    
    def test_hextree_avg_pooling(self):
        htree = get_hextree(0)
        depth = 5

        nnum_nempty = htree.nnum_nempty[depth]
        data0 = torch.zeros(nnum_nempty, 100)
        data1 = torch.ones(nnum_nempty, 100)
        avg_pooling = HextreeAvgPool(nempty=True)
        out0 = avg_pooling(data0, htree, depth)
        out1 = avg_pooling(data1, htree, depth)
        assert out0.shape[0] == htree.nnum_nempty[depth-1]
        assert out1.shape[0] == htree.nnum_nempty[depth-1]
        assert torch.all(out0 == 0)
        assert torch.all(out1 == 1)

        nnum = htree.nnum[depth]
        data = torch.rand(nnum, 100)
        avg_pooling = HextreeAvgPool(nempty=False)
        out = avg_pooling(data, htree, depth)
        assert out.shape[0] == htree.nnum[depth-1]

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    unittest.main()