import os
import torch
import unittest
import sys

from utils import get_hextree
sys.path.append('..')
# from modules import HextreeMaxPool, HextreeMaxUnpool, HextreeAvgPoolXYZ, HextreeAvgUnpoolXYZ, HextreeMaxPoolXYZ, HextreeMaxUnpoolXYZ
from modules import HextreeAvgPoolXYZ, HextreeAvgUnpoolXYZ, HextreeWeightedPoolXYZ

class HextreePoolTest(unittest.TestCase):
    # only dimension check, no value check
    # def test_hextree_max_pooling(self):
    #     htree = get_hextree(data_path='../data/points.npy', label_path='../data/semantic.npy', video_id=0)
    #     depth = 5

    #     nnum_nempty = htree.nnum_nempty[depth]
    #     data = torch.rand(nnum_nempty, 100)
    #     max_pooling = HextreeMaxPool(nempty=True, return_indices=True)
    #     out, indices = max_pooling(data, htree, depth)
    #     assert out.shape[0] == htree.nnum_nempty[depth-1]

    #     max_unpooling = HextreeMaxUnpool(nempty=True)
    #     out = max_unpooling.forward(out, indices, htree, depth-1)
    #     assert out.shape[0] == htree.nnum_nempty[depth]

    #     nnum = htree.nnum[depth]
    #     data = torch.rand(nnum, 100)
    #     max_pooling = HextreeMaxPool(nempty=False, return_indices=True)
    #     out, indices = max_pooling(data, htree, depth)
    #     assert out.shape[0] == htree.nnum[depth-1]

    #     max_unpooling = HextreeMaxUnpool(nempty=False)
    #     out = max_unpooling.forward(out, indices, htree, depth-1)
    #     assert out.shape[0] == htree.nnum[depth]
    
    # only dimension check, no value check
    def test_hextree_avg_pooling_xyz(self):
        htree = get_hextree(data_path='../../points.npy', label_path='../../semantic.npy', video_id=0)

        nnum_nempty = htree.nnum_nempty[-1]
        data = torch.rand(nnum_nempty, 100)
        Pool = HextreeAvgPoolXYZ()
        Unpool = HextreeAvgUnpoolXYZ()
        dims = [None] * htree.depth
        for d in range(htree.depth, 0, -1):
            dims[d-1] = len(data)
            data = Pool(data, htree, from_depth=d)
        print(dims)
        for d in range(0, htree.depth):
            data = Unpool(data, htree, from_depth=d)
            self.assertTrue(len(data) == dims[d])
        data = Pool(data, htree, from_depth=htree.depth, to_depth=0)
        for d in range(0, htree.depth):
            data = Unpool(data, htree, from_depth=d)
            self.assertTrue(len(data) == dims[d])
        data2 = data[:]
        for d in range(htree.depth, 3, -1):
            dims[d-1] = len(data)
            data = Pool(data, htree, from_depth=d)
        data = Unpool(data, htree, from_depth=3, to_depth=htree.depth)
        self.assertTrue(torch.allclose(data, data2))

        Pool = HextreeWeightedPoolXYZ(in_channels=100, out_channels=200, from_depth=htree.depth, to_depth=3)
        data = Pool.forward(data, htree)
        print(data.shape)

    
    # only dimension check, no value check
    # def test_hextree_max_pooling_xyz(self):
    #     htree = get_hextree(data_path='../data/points.npy', label_path='../data/semantic.npy', video_id=0)

    #     nnum_nempty = htree.nnum_nempty[-1]
    #     data = torch.rand(nnum_nempty, 100)
    #     Pool = HextreeMaxPoolXYZ(return_indices=True)
    #     Unpool = HextreeMaxUnpoolXYZ()
    #     indices = [None] * htree.depth
    #     dims = [None] * htree.depth
    #     for d in range(htree.depth, 0, -1):
    #         dims[d-1] = len(data)
    #         data, indices[d-1] = Pool(data, htree, depth=d)
    #     for d in range(0, htree.depth):
    #         data = Unpool(data, indices[d], htree, depth=d)
    #         self.assertTrue(len(data) == dims[d])


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    unittest.main()