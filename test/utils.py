# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import numpy as np
import sys
sys.path.append('..')
import hextree as ht
from test_data import read_data 


def get_points(id):
    points = read_data(save_raw=False)[id][:8*8192]
    return points


def get_hextree(id):
    point_cloud = get_points(id)
    hextree = ht.Hextree(depth=8,full_depth=1)
    hextree.build_hextree(point_cloud)
    return hextree


def get_batch_hextree(device='cpu'):
    hextree1 = get_hextree(0).to(device)
    hextree2 = get_hextree(1).to(device)
    hextree = ht.merge_hextrees([hextree1, hextree2])
    hextree.construct_all_neigh()
    return hextree
