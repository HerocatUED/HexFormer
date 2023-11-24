import torch
from torch_scatter import scatter_add, scatter_max
from hextree import Hextree

from hextree.utils import meshgrid

from .hextree_pad import hextree_pad, hextree_depad


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
    idx = htree.scatter_idx[depth]
    count = htree.masked_counts[depth]
    out = scatter_add(dim=0, index=idx, src=data) / count.unsqueeze(1)
    return out


def hextree_avg_unpool_xyz(data: torch.Tensor, htree: Hextree, depth:int):
    r''' Performs hextree average unpooling with kernel size 2 and stride 2
    on xyz axes.

    Args:
        data (torch.Tensor): The input tensor to be pooled, must corresponds to non-empty nodes 
        htree (Hextree): The corresponding hextree
        depth (int): The corresponding depth of hextree. After each unpooling, 
            the depth is increased by 1. Note that 
            :func:`hextree_avg_pool_xyz(depth=d)` matches 
            :func:`hextree_avg_unpool_xyz(depth=d-1)`
    '''
    idx = htree.scatter_idx[depth+1]
    out = data[idx]
    return out


def hextree_max_pool_xyz(data: torch.Tensor, htree: Hextree, depth: int, 
                         return_indices: bool = False):
    r''' Performs hextree max pooling with kernel size 2 and stride 2
    on xyz axes.

    Args:
        data (torch.Tensor): The input tensor to be pooled, must corresponds to non-empty nodes 
        htree (Hextree): The corresponding hextree
        depth (int): The corresponding depth of hextree. After each pooling, 
            the depth is decreased by 1. Note that 
            :func:`hextree_max_pool_xyz(depth=d)` matches 
            :func:`hextree_max_unpool_xyz(depth=d-1)`
    '''
    idx = htree.scatter_idx[depth]
    out, indices = scatter_max(dim=0, index=idx, src=data)
    if return_indices:
        return out, indices
    return out


def hextree_max_unpool_xyz(data: torch.Tensor, indices: torch.Tensor, 
                           htree: Hextree, depth: int):
    r''' Performs hextree max unpooling with kernel size 2 and stride 2
    on xyz axes.

    Args:
        data (torch.Tensor): The input tensor to be pooled, must corresponds to non-empty nodes 
        indices (torch.Tensor): The indices returned by :func:`hextree_max_pool_xyz`. The
            depth of :attr:`indices` is larger by 1 than :attr:`data`.
        htree (Hextree): The corresponding hextree
        depth (int): The corresponding depth of hextree. After each unpooling, 
            the depth is increased by 1. Note that 
            :func:`hextree_max_pool_xyz(depth=d)` matches 
            :func:`hextree_max_unpool_xyz(depth=d-1)`
    '''
    idx = htree.scatter_idx[depth+1]
    out = torch.zeros_like(data[idx])
    num, channel = data.shape
    i = torch.arange(num, device=indices.device, dtype=indices.dtype)
    k = torch.arange(channel, device=indices.device, dtype=indices.dtype)
    _, k = meshgrid(i, k, indexing='ij')
    out[indices, k] = data
    return out


def hextree_max_pool(data: torch.Tensor, hextree: Hextree, depth: int,
                     nempty: bool = False, return_indices: bool = False):
    r''' Performs hextree max pooling with kernel size 2 and stride 2. We do not recommand 
    using this function since it will compress t axis. Use :func:`hextree_max_pool_xyz` instead

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


class HextreeMaxPool(torch.nn.Module):
    r''' Performs hextree max pooling.

    Please refer to :func:`hextree_max_pool` for details.
    '''
    
    def __init__(self, nempty: bool = False, return_indices: bool = False):
        super().__init__()
        self.nempty = nempty
        self.return_indices = return_indices

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        return hextree_max_pool(data, hextree, depth, self.nempty, self.return_indices)


class HextreeMaxUnpool(torch.nn.Module):
    r''' Performs hextree max unpooling.

    Please refer to :func:`hextree_max_unpool` for details.
    '''

    def __init__(self, nempty: bool = False):
        super().__init__()
        self.nempty = nempty

    def forward(self, data: torch.Tensor, indices: torch.Tensor, hextree: Hextree,
                depth: int):
        return hextree_max_unpool(data, indices, hextree, depth, self.nempty)


class HextreeAvgPoolXYZ(torch.nn.Module):
    r''' Performs hextree average pooling on xyz axes.

    Please refer to :func:`hextree_avg_pool_xyz` for details.
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        return hextree_avg_pool_xyz(data, hextree, depth)


class HextreeAvgUnpoolXYZ(torch.nn.Module):
    r''' Performs hextree average pooling.

    Please refer to :func:`hextree_avg_unpool_xyz` for details.
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        return hextree_avg_unpool_xyz(data, hextree, depth)
    

class HextreeMaxPoolXYZ(torch.nn.Module):
    r''' Performs hextree average pooling on xyz axes.

    Please refer to :func:`hextree_max_pool_xyz` for details.
    '''
    def __init__(self, return_indices: bool = False):
        super().__init__()
        self.return_indices = return_indices
        
    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        return hextree_max_pool_xyz(data, hextree, depth, self.return_indices)


class HextreeMaxUnpoolXYZ(torch.nn.Module):
    r''' Performs hextree average pooling.

    Please refer to :func:`hextree_max_unpool_xyz` for details.
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, data: torch.Tensor, indices: torch.Tensor, hextree: Hextree, 
                depth: int):
        return hextree_max_unpool_xyz(data, indices, hextree, depth)