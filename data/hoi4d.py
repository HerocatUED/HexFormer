# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# HexFormer version modified by Xiang Wang
# --------------------------------------------------------


import torch
import numpy as np

from hextree import Points, merge_points
from thsolver import Dataset
from typing import List

from .utils import ReadFile, Transform


def align_z(points: Points):
    points.points[:, 3] -= points.points[:, 3].min()
    return points


def rand_crop(points: Points, max_npt: int):
    r''' Keeps `max_npt` pts at most centered by a radomly chosen pts. 
    '''

    pts = points.points
    npt = points.npt
    crop_mask = torch.ones(npt, dtype=torch.bool)
    if npt > max_npt:
        rand_idx = torch.randint(low=0, high=npt, size=(1,))
        sort_idx = torch.argsort(torch.sum((pts - pts[rand_idx])**2, 1))
        crop_idx = sort_idx[max_npt:]
        crop_mask[crop_idx] = False
        points = points[crop_mask]
    return points, crop_mask


class HOI4DTransform(Transform):

    def __init__(self, flags):
        super().__init__(flags.depth, flags.full_depth, flags.distort)

        # The `self.scale_factor` is used to normalize the input point cloud to the
        # range of [-1, 1]. If this parameter is modified, the `self.elastic_params`
        # and the `jittor` in the data augmentation should be scaled accordingly.
        # self.scale_factor = 5.12
        # depth 9: voxel size 2cm
        # depth 10: voxel size 2cm; 
        # depth 11: voxel size 1cm
        self.scale_factor = 10.24

    def __call__(self, sample, idx=None):
        # construct and normalize points
        pcds = Points(points=torch.from_numpy(sample['points']),
                      labels=torch.from_numpy(sample['labels']))
        bbmin, bbmax = pcds.bbox_xyz()
        pcds.normalize_xyz(bbmin, bbmax)

        # transform including rotatation, translation, scaling, and flipping
        output = self.transform(pcds, idx)   # points and inbox_mask
        points, inbox_mask = output['points'], output['inbox_mask']

        # random crop
        if self.distort:
            raise NotImplementedError
            max_npt = self.flags.max_npt if self.flags.max_npt > 0 else points.npt
            max_npt = min(max_npt, int(points.npt * self.flags.crop_ratio))
            points, crop_mask = rand_crop(points, max_npt)
            inbox_mask[inbox_mask.clone()] = crop_mask   # update inbox_mask

        # align z
        points = align_z(points)
        return {'points': points, 'inbox_mask': inbox_mask}


# def apply_cutmix(points: List[Points], cutmix: float):
#     if cutmix <= 0:
#         return points

#     batch_size = len(points)
#     outputs = [None] * batch_size
#     for i in range(batch_size):
#         j = (i + 1) % batch_size
#         points_a = points[i]
#         points_b = points[j]

#         npt_a = points_a.points.shape[0]
#         npt_b = points_b.points.shape[0]
#         na = int(cutmix * npt_a)
#         nb = int((1 - cutmix) * npt_b)

#         rand_idx = torch.randint(0, npt_a, size=(1,))
#         rand_pts = points_a.points[rand_idx]
#         dist_a, idx_a = torch.sort(
#             torch.sum((points_a.points - rand_pts)**2, 1))
#         cut_a = idx_a[:na]

#         dist_b = torch.sum((points_b.points - rand_pts)**2, 1) - dist_a[na]
#         mask_b = dist_b < 0
#         dist_b[mask_b] += 1.0e3
#         dist_b, idx_b = torch.sort(dist_b)
#         cut_b = idx_b[:nb]

#         outputs[i] = merge_points(
#             [points_a[cut_a], points_b[cut_b]], update_batch_info=False)
#     return outputs


class CollateBatch:

    def __init__(self, cutmix: int = 0.5):
        super().__init__()
        self.cutmix = cutmix

    def __call__(self, batch: list):
        assert type(batch) == list

        # a list of dicts -> a dict of lists
        outputs = {key: [b[key] for b in batch] for key in batch[0].keys()}

        # apply cutmix
        points = outputs['points']
        if self.cutmix > 0:  # and torch.rand(1) > 0.3:
            raise NotImplementedError
            points = apply_cutmix(points, self.cutmix)
        outputs['points'] = points
        return outputs


def get_hoi4d_seg_dataset(flags):
    transform = HOI4DTransform(flags)
    read_file = ReadFile(has_normal=False, has_color=False, has_label=True)
    collate_batch = CollateBatch(flags.cutmix)

    dataset = Dataset(flags.location, flags.filelist,
                      transform, read_file=read_file)
    return dataset, collate_batch
