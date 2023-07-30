# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# Hextree modified by Ruihuan Wang
# --------------------------------------------------------

import torch 
import torch.nn.functional as F
from typing import Union, List

import sys 
sys.path.append('..')
from utils import  meshgrid, scatter_add, cumsum, trunc_div
from points import Points
from shuffled_key import txyz2key, key2txyz


class Hextree:
    r''' Builds an hextree from an input pointcloud.

    Args:
        depth (int): The hextree depth.
        full_depth (int): The hextree layers with a depth small than
            :attr:`full_depth` are forced to be full.
        batch_size (int): The hextree batch size.
        device (torch.device or str): Choose from :obj:`cpu` and :obj:`gpu`.
            (default: :obj:`cpu`)

    .. note::
        The hextree data structure requires that if an hextree node has children nodes,
        the number of children nodes is exactly 16, in which some of the nodes are
        empty and some nodes are non-empty. The properties of an hextree, including
        :obj:`keys`, :obj:`children` and :obj:`neighs`, contain both non-empty and
        empty nodes, and other properties, including :obj:`features`, :obj:`normals`
        and :obj:`points`, contain only non-empty nodes.

    .. note::
        The point cloud must be in range :obj:`[-1, 1]`.
    '''