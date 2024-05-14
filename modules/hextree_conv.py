import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import List

from hextree import Hextree, key2txyz, txyz2key
from ocnn.octree import Octree
from ocnn.nn import OctreeConv, OctreeDeconv, octree_align


class TConv1(torch.nn.Module):
    r"""Convolution on t-dimension"""
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int = 3,
        nempty: bool = False,
        max_buffer: int = 20000):
        
        assert nempty == True, "nempty hardcode"
        assert kernel_size > 1, "TConv with only one frame is meaningless."
        
        self.kernel_size = kernel_size
        self.nempty = nempty
        weight = torch.randn((kernel_size, in_channels, out_channels))
        self.weights = nn.Parameter(weight, requires_grad=True)
            
    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        result = []
        data_batch = self.split(data, hextree, depth)
        for batch in data_batch:
            result += self.perform_conv(batch, hextree)
        result = torch.stack(result)
        assert data.size(0) == result.size(0), "Shape can not be modified!"
        return result
    
    def split(self, data: torch.Tensor, hextree: Hextree, depth: int):
        r"""
        split data frame by frame
        
        return: data_batch in form [[batch_1], [batch_2], ...], 
        with each [batch_i] = [data of frame_1, data of frame_2, ...]
        """
        if self.nempty:
            npts = torch.stack([
                hextree.octree_list[i].nnum_nempty[depth] 
                 for i in range(len(hextree.octree_list))])
        else:
            npts = torch.stack([
                hextree.octree_list[i].nnum[depth] 
                 for i in range(len(hextree.octree_list))])
        
        nnum_cum_nempty = torch.cumsum(npts, dim=0)
        data_batch = [data[nnum_cum_nempty[i-1] if i>0 else 0: nnum_cum_nempty[i]] 
                 for i in range(len(nnum_cum_nempty))]
        octree_cum = torch.cumsum(torch.stack(hextree.octree_num), dim=0)
        data_batch = [data_batch[octree_cum[i-1] if i>0 else 0: octree_cum[i]] 
                 for i in range(len(octree_cum))]        
        return data_batch
    
    def perform_conv(self, batch: List[torch.Tensor], octrees: List[Octree]):
        # TODO
        for i in range(1, len(batch)):
            src = torch.zeros((i+1, *batch[i].size))
            for j in range(i):
                src[j] = octree_align()
            batch[i] = 1# conv
        return torch.stack(batch)
    

class TConv(torch.nn.Module):
    r"""Convolution on t-dimension"""
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int = 3,
        nempty: bool = False):
        
        assert nempty == True, "nempty hardcode"
        assert kernel_size > 1, "TConv with only one frame is meaningless."
        
        self.kernel_size = kernel_size
        self.nempty = nempty
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
            
    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        datas = [data.unsqueeze(2)]
        key = hextree.key(depth, self.nempty)
        t, x, y, z, b = key2txyz(key, depth)
        for i in range(1, self.kernel_size):
            ti = t - i
            ti[ti<0] = 255 # ti not exist, padding with 0
            key_ti = txyz2key(ti, x, y, z, b) 
            datas.append(self.search_value(data, key, key_ti).unsqueeze(2))
        out = torch.stack(datas, dim=2)
        out = self.conv(out).squeeze(2)
        return out
    
    def search_value(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor):
        r''' Searches values according to sorted shuffled keys.

        Args:
            value (torch.Tensor): The input tensor with shape (N, C).
            key (torch.Tensor): The key tensor corresponds to :attr:`value` with shape 
                (N,), which contains sorted shuffled keys of an octree.
            query (torch.Tensor): The query tensor, which also contains shuffled keys.
        '''

        # deal with out-of-bound queries, the indices of these queries
        # returned by torch.searchsorted equal to `key.shape[0]`
        out_of_bound = query > key[-1]

        # search
        idx = torch.searchsorted(key, query)
        idx[out_of_bound] = -1   # to avoid overflow when executing the following line
        found = key[idx] == query

        # assign the found value to the output
        out = torch.zeros(query.shape[0], value.shape[1], device=value.device)
        out[found] = value[idx[found]]
        return out
                


class HextreeConv(torch.nn.Module):
    r"""Convolution, frame by frame with OctreeConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
        direct_method: bool = False,
        use_bias: bool = True,
        max_buffer: int = 200000000
    ):
        super().__init__()
        assert nempty == True, "nempty hardcode"
        self.conv = OctreeConv(in_channels, out_channels, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)
        self.down = stride == 2

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        data = data[hextree.hex2oct_nempty[depth]]
        data = self.conv(data, hextree.octrees, depth)
        if self.down:
            data = data[hextree.oct2hex_nempty[depth - 1]]
        else:
            data = data[hextree.oct2hex_nempty[depth]]
        return data


class HextreeDeconv(torch.nn.Module):
    r"""Convolution, frame by frame with OctreeConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
        direct_method: bool = False,
        use_bias: bool = True,
        max_buffer: int = 200000000
    ):
        super().__init__()
        assert nempty == True, "nempty hardcode"
        self.deconv = OctreeDeconv(in_channels, out_channels, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        data = data[hextree.hex2oct_nempty[depth]]
        data = self.deconv(data, hextree.octrees, depth)
        data = data[hextree.oct2hex_nempty[depth + 1]]
        return data


class HextreeConvBn(torch.nn.Module):
    r"""A sequence of :class:`HextreeConv` and :obj:`BatchNorm`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
        direct_method: bool = False,
        use_bias: bool = False,
        max_buffer: int = 200000000
    ):
        super().__init__()
        self.conv = HextreeConv(in_channels, out_channels, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        out = self.conv(data, hextree, depth)
        out = self.bn(out)
        return out


class HextreeConvBnRelu(torch.nn.Module):
    r"""A sequence of :class:`HextreeConv`, :obj:`BatchNorm`, and :obj:`Relu`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
        direct_method: bool = False,
        use_bias: bool = False,
        max_buffer: int = 200000000
    ):
        super().__init__()
        self.conv = HextreeConv(in_channels, out_channels, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        out = self.conv(data, hextree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out


class HextreeDeconvBn(torch.nn.Module):
    r"""A sequence of :class:`HextreeDeconv` and :obj:`BatchNorm`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
        direct_method: bool = False,
        use_bias: bool = False,
        max_buffer: int = 200000000
    ):
        super().__init__()
        self.deconv = HextreeDeconv(in_channels, out_channels, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        out = self.deconv(data, hextree, depth)
        out = self.bn(out)
        return out


class HextreeDeconvBnRelu(torch.nn.Module):
    r"""A sequence of :class:`HextreeDeconv`, :obj:`BatchNorm`, and :obj:`Relu`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
        direct_method: bool = False,
        use_bias: bool = False,
        max_buffer: int = 200000000
    ):
        super().__init__()
        self.deconv = HextreeDeconv(in_channels, out_channels, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        out = self.deconv(data, hextree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out
