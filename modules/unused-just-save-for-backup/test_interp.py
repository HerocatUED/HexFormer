# import os
# import torch
# import unittest
# import sys

# from utils import get_hextree
# sys.path.append('..')
# import hextree
# from modules import hextree_interp, HextreeUpsample, hextree_pooling, feature_init


# class HextreeInterpTest(unittest.TestCase):

#     def init_points(self):
#         points = torch.Tensor([[0, -1, -1, -1], [0, 0, 0, -1], [8, 0.0625, 0.0625, -1]])
#         normals = torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
#         features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
#         labels = torch.Tensor([[0], [2], [2]])
#         return hextree.Points(points, normals, features, labels)
    
#     def build_hextree(self, device):
#         point_cloud = self.init_points().to(device)
#         htree = hextree.Hextree(depth=3, full_depth=2, device=device)
#         htree.build_hextree(point_cloud)
#         htree = htree.to('cpu')
#         return htree
        
#     def test_hextree_interp_path(self):
#         hextree_test = self.build_hextree("cpu")
#         feature = feature_init('P')
#         out = self.upsample(feature, hextree, depth, depth_max)

import os
import torch
import numpy as np
import unittest


import sys
sys.path.append('..')
from modules.hextree_interp import hextree_nearest_upsample, hextree_nearest_pts, hextree_linear_pts
from modules import hextree_depad
import hextree


class TesOctreeInterp(unittest.TestCase):

    def init_points(self):
        points = torch.Tensor([[0, -1, -1, -1], [0, 0, 0, -1], [8, 0.0625, 0.0625, -1]])
        normals = torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
        features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
        labels = torch.Tensor([[0], [2], [2]])
        return hextree.Points(points, normals, features, labels)
    
    def build_hextree(self, device):
        point_cloud = self.init_points().to(device)
        htree = hextree.Hextree(depth=3, full_depth=2, device=device)
        htree.build_hextree(point_cloud)
        htree = htree.to('cpu')
        return htree


    def test_hextree_interp(self):
        hextree_T = self.build_hextree('cpu')
        print(hextree_T.nnum)

        depth = 3

        data = torch.zeros((hextree_T.nnum[-1], 3))
        
        pts = torch.Tensor([[0, -1, -1, -1, 0], [0, 0, 0, -1, 0], [8, 0.0625, 0.0625, -1, 0]])
        # linear = hextree_linear_pts(
        #     data, hextree, depth, pts, nempty=False)
        # linear_ne = hextree_linear_pts(
        #     data_ne, hextree, depth, pts, nempty=True)
        near = hextree_nearest_pts(
            data, hextree_T, depth, pts, nempty=False)

        # self.assertTrue(np.allclose(linear.numpy(), test['linear'], atol=1.e-6))
        # self.assertTrue(np.allclose(linear_ne.numpy(), test['linear_ne'], atol=1e-6))  # noqa
        # self.assertTrue(np.allclose(near.numpy(), test['near'], atol=1e-6))
        # self.assertTrue(np.allclose(near_ne.numpy(), test['near_ne'], atol=1e-6))
        
        print(near)
        print('keys:')
        for i, key in enumerate(hextree_T.key(-1, nempty=True)):
            print(f'{format(key[1].item(), "0>12b")}')
            
            pts2key = hextree.txyz2key(pts[i][0], pts[i][1], pts[i][2], pts[i][3], pts[i][4], 3)
            print(f'{format(pts2key[1].item(), "0>12b")}')
            pcd = hextree.key2txyz(key.cpu(), 3)
            print('convert')
            print(pcd)
            print('truth')
            print(hextree_T.points[-1][i])

    # def test_hextree_nearest_upsample(self):
    #     depth = 4
    #     depth_out = 5
    #     hextree = self.build_hextree()

    #     # test case: nempty=False
    #     nnum = hextree.nnum[depth]
    #     data = torch.rand(nnum, 4)
    #     out = hextree_nearest_upsample(data, hextree, depth=depth, nempty=False)

    #     xyzb = hextree.xyzb(depth_out, nempty=False)
    #     pts = torch.stack(xyzb, dim=1)
    #     pts[:, :3] = (pts[:, :3] + 0.5) * 0.5
    #     out_ref = hextree_nearest_pts(
    #         data, hextree, depth, pts, nempty=False, bound_check=True)
    #     self.assertTrue(np.array_equal(out.numpy(), out_ref.numpy()))

    #     # test case: nempty=False
    #     nnum = hextree.nnum_nempty[depth]
    #     data = torch.rand(nnum, 4)
    #     out = hextree_nearest_upsample(data, hextree, depth=depth, nempty=True)

    #     xyzb = hextree.xyzb(depth_out, nempty=True)
    #     pts = torch.stack(xyzb, dim=1)
    #     pts[:, :3] = (pts[:, :3] + 0.5) * 0.5
    #     out_ref = hextree_nearest_pts(
    #         data, hextree, depth, pts, nempty=True, bound_check=True)
    #     self.assertTrue(np.array_equal(out.numpy(), out_ref.numpy()))



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    unittest.main()