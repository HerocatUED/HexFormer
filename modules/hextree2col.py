# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# Hextree version written by Xiang Wang
# --------------------------------------------------------

import torch
import torch.nn

# import sys
# sys.path.append('..')
from hextree import Hextree
from hextree.utils import scatter_add


def hextree2col(data: torch.Tensor, hextree: Hextree, depth: int,
               kernel_size: str = '3333', stride: int = 1, nempty: bool = False):
  r''' Gathers the neighboring features for convolutions.

  Args:
    data (torch.Tensor): The input data.
    hextree (Hextree): The corresponding hextree.
    depth (int): The depth of current hextree.
    kernel_size (str): The kernel shape, choose from :obj:`[3,3,3,3]`, :obj:`[3,1,1,1]`, :obj:`[1,3,1,1]`, :obj:`[1,1,3,1]`, :obj:`[1,1,1,3]`,
          :obj:`[2,2,2,2]`, :obj:`[3,3,1,1]`, :obj:`[1,3,3,1]`, and :obj:`[1,1,3,3]`.
    stride (int): The stride of neighborhoods (:obj:`1` or :obj:`2`). If the
        stride is :obj:`2`, it always returns the neighborhood of the first
        siblings, and the number of elements of output tensor is
        :obj:`hextree.nnum[depth] / 8`.
    nempty (bool): If True, only returns the neighborhoods of the non-empty
        hextree nodes.
  '''

  neigh = hextree.get_neigh(depth, kernel_size, stride, nempty)
  size = (neigh.shape[0], neigh.shape[1], data.shape[1])
  out = torch.zeros(size, dtype=data.dtype, device=data.device)
  valid = neigh >= 0
  out[valid] = data[neigh[valid]]  # (N, K, C)
  return out


def col2hextree(data: torch.Tensor, hextree: Hextree, depth: int,
               kernel_size: str = '3333', stride: int = 1, nempty: bool = False):
  r''' Scatters the convolution features to an hextree.

  Please refer to :func:`hextree2col` for the usage of function parameters.
  '''

  neigh = hextree.get_neigh(depth, kernel_size, stride, nempty)
  valid = neigh >= 0
  dim_size = hextree.nnum_nempty[depth] if nempty else hextree.nnum[depth]
  out = scatter_add(data[valid], neigh[valid], dim=0, dim_size=dim_size)
  return out
