from typing import Optional
import torch 
import torch.nn as nn
from torch_scatter import scatter_add
from hextree import Hextree, key2masked


def hextree_weighted_pooling_xyz(data: torch.Tensor, 
                                 weight: torch.Tensor, 
                                 htree: Hextree, 
                                 from_depth: int, 
                                 to_depth: Optional[int]=None,
                                 need_mean: bool=False):
    if to_depth is None:
        to_depth = from_depth - 1
    assert 0 <= to_depth < from_depth <= htree.depth, 'Depth input error!'
    assert weight.shape[0] == 8 ** (from_depth - to_depth), 'Weight shape error'

    from_keys = htree.unique_keys[from_depth]
    assert from_keys.shape[0] == data.shape[0], 'Data shape error'

    from_keys_masked, xyz_index = key2masked(from_keys, htree.depth - to_depth, need_index=True)
    for i, h in enumerate(range(htree.depth - from_depth + 1, htree.depth - to_depth + 1)):
        from_bits = (1 << (4 * h - 1)) - (1 << (4 * (h - 1)))
        xyz_index_i = (((xyz_index & from_bits) >> (4 * (h - 1))) & 0x7) << (3 * i)
        xyz_index = (xyz_index & ~from_bits) | xyz_index_i
    
    _, idx, counts = torch.unique(
            from_keys_masked, sorted=True, return_inverse=True, return_counts=True, dim=0)
    weighted_data = weight[xyz_index] * data

    out = scatter_add(dim=0, index=idx, src=weighted_data)
    if need_mean:
        out /= counts.unsqueeze(1)
    return out
    

def hextree_avg_pool_xyz(data: torch.Tensor, 
                         htree: Hextree, 
                         from_depth: int,
                         to_depth: Optional[int]=None):
    if to_depth is None:
        to_depth = from_depth - 1
    assert 0 <= to_depth < from_depth <= htree.depth, 'Depth input error!'

    weight = torch.ones(8 ** (from_depth - to_depth), 1)
    out = hextree_weighted_pooling_xyz(data, weight, htree,
                                       from_depth, to_depth, True)
    return out


def hextree_avg_unpool_xyz(data: torch.Tensor, 
                           htree: Hextree, 
                           from_depth: int,
                           to_depth: Optional[int]=None):
    if to_depth is None:
        to_depth = from_depth + 1
    assert 0 <= from_depth < to_depth <= htree.depth, 'Depth input error!'

    to_keys = htree.unique_keys[to_depth]
    from_keys = htree.unique_keys[from_depth]
    assert from_keys.shape[0] == data.shape[0], 'Data shape error'

    to_keys_masked = key2masked(to_keys, htree.depth - from_depth)
    _, idx = torch.unique(
            to_keys_masked, sorted=True, return_inverse=True, dim=0)
    
    out = data[idx]
    return out


class HextreeAvgPoolXYZ(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data: torch.Tensor, htree: Hextree,
                from_depth: int, to_depth: Optional[int]=None):
        return hextree_avg_pool_xyz(data, htree, from_depth, to_depth)


class HextreeAvgUnpoolXYZ(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data: torch.Tensor, htree: Hextree,
                from_depth: int, to_depth: Optional[int]=None):
        return hextree_avg_unpool_xyz(data, htree, from_depth, to_depth)
    

class HextreeWeightedPoolXYZ(torch.nn.Module):
    def __init__(self, from_depth: int, to_depth: Optional[int]=None):
        super().__init__()
        if to_depth is None:
            to_depth = from_depth - 1
        self.from_depth = from_depth
        self.to_depth = to_depth
        weight = torch.randn(8 ** (from_depth - to_depth), 1)
        self.weight = nn.Parameter(weight, requires_grad=True)
    
    def forward(self, data: torch.Tensor, htree: Hextree, need_mean: bool=False):
        return hextree_weighted_pooling_xyz(data, self.weight, htree, 
                                            self.from_depth, self.to_depth, need_mean)