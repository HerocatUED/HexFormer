# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# Modified by Ruihuan Wang
# --------------------------------------------------------

import torch
from typing import List
from hextree import Hextree

from hextree.utils import meshgrid, scatter_add, resize_with_last_val, list2str

from .hextree_pad import hextree_pad, hextree_depad


def hextree_max_pool(data: torch.Tensor, hextree: Hextree, depth: int,
                     nempty: bool = False, return_indices: bool = False):
    r''' Performs hextree max pooling with kernel size 2 and stride 2.

    Args:
        data (torch.Tensor): The input tensor.
        hextree (Hextree): The corresponding hextree.
        depth (int): The depth of current hextree. After pooling, the corresponding
            depth decreased by 1.
        nempty (bool): If True, :attr:`data` contains only features of non-empty
            hextree nodes.
        return_indices (bool): If True, returns the indices, which can be used in
            :func:`hextree_max_unpool`.
    '''
    if nempty:
        data = hextree_pad(data, hextree, depth, float('-inf'))
    data = data.view(-1, 16, data.shape[1])
    out, indices = data.max(dim=1)
    if not nempty:
        out = hextree_pad(out, hextree, depth-1)
    return (out, indices) if return_indices else out


def hextree_max_unpool(data: torch.Tensor, indices: torch.Tensor, hextree: Hextree,
                       depth: int, nempty: bool = False):
    r''' Performs hextree max unpooling.

    Args:
        data (torch.Tensor): The input tensor.
        indices (torch.Tensor): The indices returned by :func:`hextree_max_pool`. The
            depth of :attr:`indices` is larger by 1 than :attr:`data`.
        hextree (Hextree): The corresponding hextree.
        depth (int): The depth of current data. After unpooling, the corresponding
            depth increases by 1.
    '''

    if not nempty:
        data = hextree_depad(data, hextree, depth)
    num, channel = data.shape 
    out = torch.zeros(num, 16, channel, dtype=data.dtype, device=data.device)
    i = torch.arange(num, dtype=indices.dtype, device=indices.device)
    k = torch.arange(channel, dtype=indices.dtype, device=indices.device)
    i, k = meshgrid(i, k, indexing='ij')
    out[i, indices, k] = data
    out = out.view(-1, channel)
    if nempty:
        out = hextree_depad(out, hextree, depth+1)
    return out


def hextree_avg_pool(data: torch.Tensor, hextree: Hextree, depth: int,
                     nempty: bool = False):
    r''' Performs hextree average pooling with kernel size 2 and stride 2.

    Args:
        data (torch.Tensor): The input tensor.
        hextree (Hextree): The corresponding hextree.
        depth (int): The depth of current hextree.
        nempty (bool): If True, :attr:`data` contains only features of non-empty
            hextree nodes.
    '''
    if nempty:
        data = hextree_pad(data, hextree, depth, 0.0)
    data = data.view(-1, 16, data.shape[1])
    non_zero_count = torch.count_nonzero(data, dim=1).type_as(data).to(data.device) + 1e-8
    out = data.sum(dim=1) / non_zero_count
    if not nempty:
        out = hextree_pad(out, hextree, depth-1)
    return out


class HextreePoolBase(torch.nn.Module):
    r''' The base class for hextree-based pooling.
    '''

    def __init__(self, kernel_size: List[int], stride: int, nempty: bool = False):
        super().__init__()
        self.kernel_size = resize_with_last_val(kernel_size)
        self.kernel = list2str(self.kernel_size)
        self.stride = stride
        self.nempty = nempty

    def extra_repr(self) -> str:
        return ('kernel_size={}, stride={}, nempty={}').format(
            self.kernel_size, self.stride, self.nempty)
    

class HextreeMaxPool(HextreePoolBase):
    r''' Performs hextree max pooling.

    Please refer to :func:`hextree_max_pool` for details.
    '''
    
    def __init__(self, nempty: bool = False, return_indices: bool = False):
        super().__init__(kernel_size=[2], stride=2, nempty=nempty)
        self.return_indices = return_indices

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        return hextree_max_pool(data, hextree, depth, self.nempty, self.return_indices)


class HextreeMaxUnpool(HextreePoolBase):
    r''' Performs hextree max unpooling.

    Please refer to :func:`hextree_max_unpool` for details.
    '''

    def __init__(self, nempty: bool = False):
        super().__init__(kernel_size=[2], stride=2, nempty=nempty)

    def forward(self, data: torch.Tensor, indices: torch.Tensor, hextree: Hextree,
                depth: int):
        return hextree_max_unpool(data, indices, hextree, depth, self.nempty)


class HextreeAvgPool(HextreePoolBase):
  r''' Performs hextree average pooling.

  Please refer to :func:`hextree_avg_pool` for details.
  '''
  def __init__(self, nempty: bool = False):
      super().__init__(kernel_size=[2], stride=2, nempty=nempty)
      
  def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
    r''''''

    return hextree_avg_pool(data, hextree, depth, self.nempty)