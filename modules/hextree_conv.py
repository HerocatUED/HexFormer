import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import List

from hextree import Hextree, key2txyz, txyz2key
from ocnn.nn import OctreeConv, OctreeDeconv

    
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
            ti[ti < 0] = 0 # ti not exist, padding with t0
            key_ti = txyz2key(ti, x, y, z, b) 
            datas.append(self.search_value(data, key, key_ti).unsqueeze(2))
        data = torch.stack(datas, dim=2)
        data = self.conv(data).squeeze(2)
        return data
    
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
        max_buffer: int = 200000000,
        # t_kernel_size: int = 3,
    ):
        super().__init__()
        assert nempty == True, "nempty hardcode"
        self.conv = OctreeConv(in_channels, out_channels, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)
        # self.tconv = TConv(in_channels, out_channels, t_kernel_size, nempty)
        self.down = stride == 2

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        data = data[hextree.hex2oct_nempty[depth]]
        data = self.conv(data, hextree.octrees, depth)
        if self.down:
            data = data[hextree.oct2hex_nempty[depth - 1]]
        else:
            data = data[hextree.oct2hex_nempty[depth]]
        # data = self.tconv(data)
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
