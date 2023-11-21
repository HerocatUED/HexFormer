import torch
from typing import List
from hextree import Hextree

from hextree.utils import meshgrid, scatter_add, resize_with_last_val, list2str

from .hextree_pad import hextree_pad, hextree_depad

class ScatterMaskLUT:
    def __init__(self):
        device = torch.device('cpu')
        self._encode = {device: torch.tensor([
            0x7fffffffffffffff, 0x7ffffffffffffff8, 0x7fffffffffffff88, 0x7ffffffffffff888,
            0x7fffffffffff8888, 0x7ffffffffff88888, 0x7fffffffff888888, 0x7ffffffff8888888,
            0x7fffffff88888888, 0x7ffffff888888888, 0x7fffff8888888888, 0x7ffff88888888888,
            0x7fff888888888888, 0x7ff8888888888888, 0x7f88888888888888
        ], dtype=torch.int64)}

    def encode_lut(self, device=torch.device('cpu')):
        if device not in self._encode:
            cpu = torch.device('cpu')
            self._encode[device] = tuple(e.to(device)
                                         for e in self._encode[cpu])
        return self._encode[device]

_scatter_mask_lut = ScatterMaskLUT()

def hextree_avg_pool_xyz(data: torch.Tensor, htree: Hextree, depth: int):
    r''' Performs hextree average pooling with kernel size 2 and stride 2
    on xyz axes.

    Args:
        data (torch.Tensor): The input tensor to be pooled, must corresponds to non-empty nodes 
        htree (Hextree): The corresponding hextree
        depth (int): The corresponding depth of hextree. After each pooling, 
            the depth is decreased by 1. Note that 
            :func:`hextree_avg_pool_xyz(depth=d)` matches 
            :func:`hextree_avg_unpool_xyz(depth=d-1)`
    '''
    keys_masked = htree.key(-1, nempty=True)
    mask_lut = _scatter_mask_lut.encode_lut(data.device)
    for d in range(htree.depth, depth, -1):
        keys_masked = keys_masked & mask_lut[htree.depth - d + 1]
        keys_masked = torch.unique(keys_masked, dim=0)
    keys_masked = keys_masked & mask_lut[htree.depth - depth + 1]
    keys_masked, idx, count = torch.unique(
        keys_masked, sorted=True, return_inverse=True, return_counts=True, dim=0
    )
    out = scatter_add(dim=0, index=idx, src=data) / count.unsqueeze(1)
    return out

def hextree_avg_unpool_xyz(data: torch.Tensor, htree: Hextree, depth:int):
    r''' Performs hextree average unpooling with kernel size 2 and stride 2
    on xyz axes.

    Args:
        data (torch.Tensor): The input tensor to be pooled, must corresponds to non-empty nodes 
        htree (Hextree): The corresponding hextree
        depth (int): The corresponding depth of hextree. After each pooling, 
            the depth is increased by 1. Note that 
            :func:`hextree_avg_pool_xyz(depth=d)` matches 
            :func:`hextree_avg_unpool_xyz(depth=d-1)`
    '''
    keys_masked = htree.key(-1, nempty=True)
    mask_lut = _scatter_mask_lut.encode_lut(data.device)
    for d in range(htree.depth, depth+1, -1):
        keys_masked = keys_masked & mask_lut[htree.depth - d + 1]
        keys_masked = torch.unique(keys_masked, dim=0)
    keys_masked = keys_masked & mask_lut[htree.depth - depth + 1]
    keys_masked, idx = torch.unique(
        keys_masked, sorted=True, return_inverse=True, dim=0
    )
    out = data[idx]
    return out

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


class HextreeAvgPoolXYZ(HextreePoolBase):
    r''' Performs hextree average pooling on xyz axes.

    Please refer to :func:`hextree_avg_pool_xyz` for details.
    '''
    def __init__(self):
        super().__init__(kernel_size=[2], stride=2, nempty=True)
        
    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        r''''''

        return hextree_avg_pool_xyz(data, hextree, depth)
    
class HextreeAvgUnpoolXYZ(HextreePoolBase):
    r''' Performs hextree average pooling.

    Please refer to :func:`hextree_avg_unpool_xyz` for details.
    '''
    def __init__(self):
        super().__init__(kernel_size=[2], stride=2, nempty=True)
        
    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        r''''''

        return hextree_avg_unpool_xyz(data, hextree, depth)