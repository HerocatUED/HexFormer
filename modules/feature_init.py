# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# Hextree version written by Xiang Wang
# --------------------------------------------------------

import torch
from hextree import Hextree
from .hextree_pad import hextree_pad


class InputFeature(torch.nn.Module):
    r''' Returns the initial input feature stored in hexree.

    Args:
      feature (str): A string used to indicate which features to extract from the
          input hexree. If the character :obj:`N` is in :attr:`feature`, the
          normal signal is extracted (3 channels). Similarly, if :obj:`D` is in
          :attr:`feature`, the local displacement is extracted (1 channels). If
          :obj:`L` is in :attr:`feature`, the local coordinates of the averaged
          points in each hexree node is extracted (3 channels). If :attr:`P` is in
          :attr:`feature`, the global coordinates are extracted (3 channels). If
          :attr:`F` is in :attr:`feature`, other features (like colors) are
          extracted (k channels).
      nempty (bool): If false, gets the features of all hexree nodes. 
    '''

    def __init__(self, feature: str = 'NDF', nempty: bool = False):
        super().__init__()
        self.nempty = nempty
        self.feature = feature.upper()

    def forward(self, hexree: Hextree):
        r''''''

        features = list()
        depth = hexree.depth
        if 'N' in self.feature:
            features.append(hexree.normals[depth])

        if 'L' in self.feature or 'D' in self.feature:
            local_points = hexree.points[depth].frac() - 0.5

        if 'D' in self.feature:
            dis = torch.sum(
                local_points * hexree.normals[depth], dim=1, keepdim=True)
            features.append(dis)

        if 'L' in self.feature:
            features.append(local_points)

        if 'P' in self.feature:
            scale = 2 ** (1 - depth)   # normalize [0, 2^depth] -> [-1, 1]
            global_points = hexree.points[depth] * scale - 1.0
            features.append(global_points)

        if 'F' in self.feature:
            features.append(hexree.features[depth])

        out = torch.cat(features, dim=1)
        if not self.nempty:
            out = hextree_pad(out, hexree, depth)
        return out

    def extra_repr(self) -> str:
        r''''''
        return 'feature={}, nempty={}'.format(self.feature, self.nempty)
