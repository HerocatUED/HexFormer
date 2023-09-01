# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np

# import ocnn
import hextree as ht


def get_points(id, return_data=False):
    folder = os.path.dirname(__file__)
    filename = os.path.join(folder, 'data/octree/test_%03d.npz' % id)
    data = np.load(filename)

    points, normals = data['points'], data['normals']
    point_cloud = ht.Points(torch.from_numpy(points), torch.from_numpy(normals))
    return (point_cloud, data) if return_data else point_cloud


def get_hextree(id, return_data=False):
    point_cloud, data = get_points(id, return_data=True)
    hextree = ht.Hextree(data['depth'].item(), full_depth=data['full_depth'].item())
    hextree.build_hextree(point_cloud)
    return (hextree, data) if return_data else hextree


def get_batch_hextree(device='cpu'):
    hextree1 = get_hextree(4).to(device)
    hextree2 = get_hextree(5).to(device)
    hextree = ht.merge_hextrees([hextree1, hextree2])
    hextree.construct_all_neigh()
    return hextree
