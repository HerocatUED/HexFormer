import os
import torch
import unittest
import math

import sys

sys.path.append("..")
from hextree import Points, merge_points


class TestPoints(unittest.TestCase):

    def init_points(self):
        s2, s3 = 2.0**0.5 / 2.0, 3.0**0.5 / 3.0
        points = torch.Tensor([[1, 1, 2, 3], [2, -4, -5, -6]])
        normals = torch.Tensor([[s2, -s2, 0], [-s3, -s3, s3]])
        labels = torch.Tensor([[1], [2]])
        features = torch.rand(2, 4)
        return Points(points, normals, features, labels)

    def test_properties(self):
        pcd = self.init_points()

        # test npt
        self.assertTrue(pcd.npt == 2)

        # test bbox_xyz
        bbmin, bbmax = pcd.bbox_xyz()
        bbmin_gt = torch.Tensor([-4, -5, -6])
        bbmax_gt = torch.Tensor([1, 2, 3])
        self.assertTrue(torch.allclose(bbmin, bbmin_gt))
        self.assertTrue(torch.allclose(bbmax, bbmax_gt))

        # test inbox_mask
        bbmax = 2.0
        bbmin = torch.Tensor([-5, -6, -7])
        mask = pcd.inbox_mask(bbmin, bbmax)
        mask_gt = torch.Tensor([False, True])
        self.assertTrue((mask == mask_gt).all())

    def test_scale_flip(self):
        pcd = self.init_points()

        # test scale_xyz
        points_gt = torch.Tensor([[1, 0.5, 1, 1.5], [2, -2, -2.5, -3]])
        s2, s3 = 2.0**0.5 / 2.0, 3.0**0.5 / 3.0
        normals_gt = torch.Tensor([[s2, -s2, 0], [-s3, -s3, s3]])
        pcd.scale_xyz(0.5)
        self.assertTrue(torch.allclose(pcd.normals, normals_gt))
        self.assertTrue(torch.allclose(pcd.points, points_gt))

        # test scale_txyz
        scale_factor = torch.Tensor([2, 1, 4, 6])
        points_gt = torch.Tensor([[2, 0.5, 4, 9], [4, -2, -10, -18]])
        normals_gt = normals_gt / torch.Tensor([1, 4, 6])
        normals_gt /= torch.norm(normals_gt, dim=1, keepdim=True)
        pcd.scale_txyz(scale_factor)
        self.assertTrue(torch.allclose(pcd.normals, normals_gt))
        self.assertTrue(torch.allclose(pcd.points, points_gt))

        # test flip
        points_gt = torch.Tensor([[2, 0.5, -4, -9], [4, -2, 10, 18]])
        normals_gt *= torch.Tensor([1, -1, -1])
        pcd.flip("yz")
        self.assertTrue(torch.allclose(pcd.normals, normals_gt))
        self.assertTrue(torch.allclose(pcd.points, points_gt))

    def test_translate(self):
        pcd = self.init_points()

        # test translate_xyz with float
        pcd.translate_xyz(1.5)
        points_gt = torch.Tensor([[1, 2.5, 3.5, 4.5], [2, -2.5, -3.5, -4.5]])
        self.assertTrue(torch.allclose(pcd.points, points_gt))

        # test translate_txyz with 2-element tensor
        dis = torch.Tensor([2, 3])
        points_gt = torch.Tensor([[3, 5.5, 6.5, 7.5], [4, 0.5, -0.5, -1.5]])
        pcd.translate_txyz(dis)
        self.assertTrue(torch.allclose(pcd.points, points_gt))

        # test translate_txyz with 4-element tensor
        dis = torch.Tensor([-1, 0.5, -1.5, 2.5])
        points_gt = torch.Tensor([[2, 6, 5, 10], [3, 1, -2, 1]])
        pcd.translate_txyz(dis)
        self.assertTrue(torch.allclose(pcd.points, points_gt))

    def test_normalize(self):
        pcd = self.init_points()

        # test normalize_xyz, not keep shape
        pcd.flip("y")
        pcd.normalize_xyz(keep_shape=False)
        s2, s3 = 2.0**0.5 / 2.0, 3.0**0.5 / 3.0
        points_gt = torch.Tensor([[1, 1, -1, 1], [2, -1, 1, -1]])
        normals_gt = torch.Tensor([[s2, s2, 0], [-s3, s3, s3]])
        normals_gt *= torch.Tensor([5, 7, 9])
        normals_gt /= torch.norm(normals_gt, dim=1, keepdim=True)
        self.assertTrue(torch.allclose(points_gt, pcd.points))
        self.assertTrue(torch.allclose(normals_gt, pcd.normals))

        pcd = self.init_points()

        # test normalize_xyz, not keep shape
        pcd.flip("y")
        pcd.normalize_xyz(keep_shape=True)
        points_gt = torch.Tensor([[1, 5 / 9, -7 / 9, 1], [2, -5 / 9, 7 / 9, -1]])
        normals_gt = torch.Tensor([[s2, s2, 0], [-s3, s3, s3]])
        self.assertTrue(torch.allclose(points_gt, pcd.points))
        self.assertTrue(torch.allclose(normals_gt, pcd.normals))

    def test_clip(self):
        pcd = self.init_points()
        features_gt = pcd.features.clone()[0]

        minb = -3
        maxb = torch.Tensor([2, 3, 4])
        mask = pcd.clip_xyz(minb, maxb)

        points_gt = torch.Tensor([[1, 1, 2, 3]])
        s2 = 2.0**0.5 / 2.0
        normals_gt = torch.Tensor([[s2, -s2, 0]])
        labels_gt = torch.Tensor([[1]])
        mask_gt = torch.Tensor([True, False])

        self.assertTrue(torch.allclose(pcd.points, points_gt))
        self.assertTrue(torch.allclose(pcd.normals, normals_gt))
        self.assertTrue(torch.allclose(pcd.labels, labels_gt))
        self.assertTrue(torch.allclose(pcd.features, features_gt))
        self.assertTrue((mask == mask_gt).all())

    def test_orient_normal(self):
        pcd = self.init_points()

        # test orient_normal
        s2, s3 = 2.0**0.5 / 2.0, 3.0**0.5 / 3.0
        normals_gt = torch.Tensor([[s2, -s2, 0], [s3, s3, -s3]])
        pcd.orient_normal("x")
        self.assertTrue(torch.allclose(pcd.normals, normals_gt))

    def test_rotation(self):
        pcd = self.init_points()
        s2, s3 = 2.0**0.5 / 2.0, 3.0**0.5 / 3.0

        # rot x
        angle = torch.Tensor([math.pi / 2.0, 0.0, 0.0])
        normals_gt = torch.tensor([[s2, 0, -s2], [-s3, -s3, -s3]], dtype=torch.float)
        points_gt = torch.tensor([[1, 1, -3, 2], [2, -4, 6, -5]], dtype=torch.float)

        pcd.rotate(angle)
        self.assertTrue(torch.allclose(pcd.normals, normals_gt, atol=1e-6))
        self.assertTrue(torch.allclose(pcd.points, points_gt, atol=1e-6))

        # rot y
        angle = torch.Tensor([0.0, math.pi / 2.0, 0.0])
        normals_gt = torch.tensor([[-s2, 0, -s2], [-s3, -s3, s3]], dtype=torch.float)
        points_gt = torch.tensor([[1, 2, -3, -1], [2, -5, 6, 4]], dtype=torch.float)

        pcd.rotate(angle)
        self.assertTrue(torch.allclose(pcd.normals, normals_gt, atol=1e-6))
        self.assertTrue(torch.allclose(pcd.points, points_gt, atol=1e-6))

        # rot z
        angle = torch.Tensor([0.0, 0.0, math.pi / 2.0])
        normals_gt = torch.tensor([[0, -s2, -s2], [s3, -s3, s3]], dtype=torch.float)
        points_gt = torch.tensor([[1, 3, 2, -1], [2, -6, -5, 4]], dtype=torch.float)

        pcd.rotate(angle)
        self.assertTrue(torch.allclose(pcd.normals, normals_gt, atol=1e-6))
        self.assertTrue(torch.allclose(pcd.points, points_gt, atol=1e-6))

    def test_getitem(self):
        pcd = self.init_points()
        s2 = 2.0**0.5 / 2.0
        sub_pcd = pcd[0]

        features_gt = pcd.features.clone()[[0]]
        sub_pcd_gt = Points(
            torch.Tensor([[1, 1, 2, 3]]),
            torch.Tensor([[s2, -s2, 0]]),
            features_gt,
            torch.Tensor([[1]]),
        )

        self.assertTrue(torch.allclose(sub_pcd.normals, sub_pcd_gt.normals))
        self.assertTrue(torch.allclose(sub_pcd.points, sub_pcd_gt.points))
        self.assertTrue(torch.allclose(sub_pcd.labels, sub_pcd_gt.labels))
        self.assertTrue(torch.allclose(sub_pcd.features, sub_pcd_gt.features))

    def test_device(self):
        pcd = self.init_points()
        pcd.to("cuda")
        pcd.to("cpu")
        pcd.cuda()
        pcd.cpu()

    def test_merge(self):
        pcd1 = self.init_points()
        pcd2 = self.init_points()
        pcd2.translate_xyz(1)
        pcd = merge_points([pcd1, pcd2])

        features_gt1 = pcd1.features.clone()
        features_gt2 = pcd2.features.clone()
        points_gt = torch.Tensor(
            [[1, 1, 2, 3], [2, -4, -5, -6], [1, 2, 3, 4], [2, -3, -4, -5]]
        )
        s2, s3 = 2.0**0.5 / 2.0, 3.0**0.5 / 3.0
        normals_gt = torch.Tensor(
            [[s2, -s2, 0], [-s3, -s3, s3], [s2, -s2, 0], [-s3, -s3, s3]]
        )
        labels_gt = torch.Tensor([[1], [2], [1], [2]])
        features_gt = torch.cat([features_gt1, features_gt2])

        self.assertTrue(torch.allclose(pcd.points, points_gt))
        self.assertTrue(torch.allclose(pcd.normals, normals_gt))
        self.assertTrue(torch.allclose(pcd.labels, labels_gt))
        self.assertTrue(torch.allclose(pcd.features, features_gt))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    unittest.main()
