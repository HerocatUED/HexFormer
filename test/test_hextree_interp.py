import sys

sys.path.append("..")
from hextree import merge_points, Points, Hextree, key2txyz, txyz2key
from modules.hextree_interp import (
    hextree_nearest_upsample,
    hextree_nearest_pts,
    hextree_linear_pts,
)
import os
import torch
import numpy as np
import unittest


class TesOctreeInterp(unittest.TestCase):

    def init_points(self):
        points = torch.Tensor([[0, -1, -1, -1], [0, 0, 0, -1], [8, 0.0625, 0.0625, -1]])
        normals = torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
        features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
        labels = torch.Tensor([[0], [2], [2]])
        return Points(points, normals, features, labels)

    def build_hextree(self, device, depth):
        point_cloud = self.init_points().to(device)
        htree = Hextree(depth, full_depth=2, device=device)
        htree.build_hextree(point_cloud)
        htree = htree.to("cpu")
        return htree

    def test_hextree_interp(self, device: str = "cpu"):
        depth = 3
        htree = self.build_hextree(device, depth)
        # print(htree.nnum[depth], htree.nnum_nempty[depth])

        # print('keys:')
        # for i, key in enumerate(htree.key(depth, nempty=True)):
        #     print(f'{format(key.item(), "0>12b")}')
        #     pcd = key2txyz(key.cpu(), depth)
        #     print('convert from key', pcd)
        #     print('truth', htree.points[depth][i])

        data = torch.zeros((htree.nnum[depth], 5)).to(device)
        data_ne = torch.zeros((htree.nnum_nempty[depth], 5)).to(device)

        points = torch.tensor(
            [[0, -1, -1, -1], [0, 0, 0, -1], [8, 0.0625, 0.0625, -1]]
        ).to(device)
        query_pts = torch.tensor(
            [[0.0, 0, 0, 0, 0], [0, 4, 4, 0, 0], [7, 4, 4, 0, 0]]
        ).to(device)
        rand_query = torch.rand((10, 5)).to(device)

        # NOTE: We use deconv as linear interp module, so this module is unused
        # linear = hextree_linear_pts(data, htree, depth, query_pts, nempty=False)
        # linear_ne = hextree_linear_pts(data, htree, depth, query_pts, nempty=True)
        # self.assertTrue(np.allclose(linear.numpy(), np.zeros(5), atol=1.e-6))
        # self.assertTrue(np.allclose(linear_ne.numpy(), np.zeros(5), atol=1e-6))  # noqa

        near = hextree_nearest_pts(data, htree, depth, query_pts, nempty=False)
        near_ne = hextree_nearest_pts(data_ne, htree, depth, query_pts, nempty=True)

        self.assertTrue(np.allclose(near.numpy(), np.zeros(5), atol=1e-6))
        self.assertTrue(np.allclose(near_ne.numpy(), np.zeros(5), atol=1e-6))

        # random interp test
        near_rand = hextree_nearest_pts(data, htree, depth, rand_query, nempty=False)
        near_ne_rand = hextree_nearest_pts(
            data_ne, htree, depth, rand_query, nempty=True
        )

        self.assertTrue(np.allclose(near_rand.numpy(), np.zeros(5), atol=1e-6))
        self.assertTrue(np.allclose(near_ne_rand.numpy(), np.zeros(5), atol=1e-6))

    # NOTE: We use deconv as upsample module, so this module is unused

    # def test_hextree_nearest_upsample(self, device:str = 'cpu'):
    #     depth = 3
    #     depth_out = 4
    #     htree = self.build_hextree('cpu')

    #     # test case: nempty=False
    #     nnum = htree.nnum[depth]
    #     data = torch.rand(nnum, 5).to(device)
    #     out = hextree_nearest_upsample(data, htree, depth=depth, nempty=False)

    #     txyzb = htree.txyzb(depth_out, nempty=False)
    #     pts = torch.stack(txyzb, dim=1)
    #     pts[:, 1:4] = (pts[:, 1:4] + 0.5) * 0.5
    #     out_ref = hextree_nearest_pts(data, htree, depth, pts, nempty=False, bound_check=True)
    #     self.assertTrue(np.array_equal(out.numpy(), out_ref.numpy()))

    #     # test case: nempty=False
    #     nnum = htree.nnum_nempty[depth]
    #     data = torch.rand(nnum, 5).to(device)
    #     out = hextree_nearest_upsample(data, htree, depth=depth, nempty=True)

    #     txyzb = htree.txyzb(depth_out, nempty=True)
    #     pts = torch.stack(txyzb, dim=1)
    #     pts[:, 1:4] = (pts[:, 1:4] + 0.5) * 0.5
    #     out_ref = hextree_nearest_pts(data, hextree, depth, pts, nempty=True, bound_check=True)
    #     self.assertTrue(np.array_equal(out.numpy(), out_ref.numpy()))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    unittest.main()
