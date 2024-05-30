import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import List

from hextree import Hextree
from ocnn.nn import OctreeConv, OctreeDeconv


class FastConv1d(nn.Module):
    r"""input (N, in_C, kernel_size) and output (N, out_C)"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(FastConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels * kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, data: torch.Tensor):
        data = data.view(-1, self.in_channels * self.kernel_size)
        data = torch.mm(data, self.weight.t()) + self.bias
        data = data.view(-1, self.out_channels)
        return data

    
class TConv(torch.nn.Module):
    r"""Hextree based Convolution on t-dimension"""
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        nempty: bool = False,
        pad_mode: int = 2):
        super().__init__()
        r'''
        Args:
        pad_mode(int):
            0 means pad with all-zero feature
            1 means pad with t_min feature
            2 means pad with shared learnable feature
        '''
        
        assert nempty == True, "nempty hardcode"
        assert kernel_size > 1, "TConv with only one frame is meaningless."
        
        self.kernel_size = kernel_size
        self.nempty = nempty
        # self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv = FastConv1d(in_channels, out_channels, kernel_size)
        self.pad_mode = pad_mode
        self.pad_feat = None
        if pad_mode == 2:
            self.pad_feat = nn.Parameter(torch.randn(in_channels))
    
    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        datas = [data]
        key = hextree.key(depth, self.nempty)
        t = key & 255
        bxyz = (key >> 8) << 8
        for i in range(1, self.kernel_size):
            ti = t - i
            # padding
            ti[ti < 0] = 0 if self.pad_mode == 1 else 255
            key_ti = bxyz | ti 
            datas.append(self.search_value(data, key, key_ti, self.pad_feat))
        data = torch.stack(datas, dim=2)
        data = self.conv(data)
        return data
    
    def search_value(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, pad_feat: torch.Tensor = None):
        r''' Searches values according to sorted shuffled keys.

        Args:
            value (torch.Tensor): The input tensor with shape (N, C).
            key (torch.Tensor): The key tensor corresponds to :attr:`value` with shape 
                (N,), which contains sorted shuffled keys of an octree.
            query (torch.Tensor): The query tensor, which also contains shuffled keys.
            pad_feat (torch.Tensor): If not none, pad the out-of-bound queries with pad_feat tensor.
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
        if pad_feat is not None: 
            out[torch.logical_not(found)] = pad_feat
        return out
                

class HextreeConv(torch.nn.Module):
    r"""Hextree based Convolution"""

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
        t_kernel_size: int = 3,
    ):
        super().__init__()
        assert nempty == True, "nempty hardcode"
        self.conv = OctreeConv(in_channels, out_channels, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)
        # self.tconv = TConv(out_channels, out_channels, t_kernel_size, 1, nempty)
        # self.bn = torch.nn.BatchNorm1d(out_channels)
        self.down = stride == 2

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        data = data[hextree.hex2oct_nempty[depth]]
        data = self.conv(data, hextree.octrees, depth)
        if self.down:
            data = data[hextree.oct2hex_nempty[depth - 1]]
            # data = self.bn(data)
            # data = self.tconv(data, hextree, depth - 1)
        else:
            data = data[hextree.oct2hex_nempty[depth]]
            # data = self.bn(data)
            # data = self.tconv(data, hextree, depth)
        return data


class HextreeDeconv(torch.nn.Module):
    r"""Hextree based DeConvolution"""

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
        t_kernel_size: int = 3,
    ):
        super().__init__()
        assert nempty == True, "nempty hardcode"
        self.deconv = OctreeDeconv(in_channels, out_channels, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)
        # self.tconv = TConv(out_channels, out_channels, t_kernel_size, 1, nempty)
        # self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        data = data[hextree.hex2oct_nempty[depth]]
        data = self.deconv(data, hextree.octrees, depth)
        data = data[hextree.oct2hex_nempty[depth + 1]]
        # data = self.bn(data)
        # data = self.tconv(data, hextree, depth + 1)
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
        self.act = torch.nn.LeakyReLU(inplace=True)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        out = self.conv(data, hextree, depth)
        out = self.bn(out)
        out = self.act(out)
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
        self.act = torch.nn.LeakyReLU(inplace=True)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        out = self.deconv(data, hextree, depth)
        out = self.bn(out)
        out = self.act(out)
        return out


class HextreeGroupConv(torch.nn.Module):
    r"""Hextree based GroupConvolution. NOTE:in_channel==out_channel"""
    def __init__(
        self,
        channels: int,
        group_size: int = 32,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
        direct_method: bool = False,
        use_bias: bool = True,
        max_buffer: int = 200000000,
        t_kernel_size: int = 3,
        ):
        super().__init__()
        
        assert channels % group_size == 0
        
        self.group_size = group_size
        self.group_num = channels // group_size
        self.convs = torch.nn.ModuleList(
            OctreeConv(group_size, group_size, kernel_size, 
            stride, nempty, use_bias, direct_method, max_buffer)
            for _ in range(self.group_num))
        self.tconv = TConv(channels, channels, t_kernel_size, 1, nempty)
        self.bn1 = nn.GroupNorm(self.group_num, channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        # data (N, C), C = k * group_size
        assert data.size(dim = 1) % self.group_size == 0
        data = data[hextree.hex2oct_nempty[depth]]
        for i in range(self.group_num):
            data[:, self.group_size*i:self.group_size*(i+1)] = \
                self.convs[i](data[:, self.group_size*i:self.group_size*(i+1)], hextree.octrees, depth)
        data = data[hextree.oct2hex_nempty[depth]]
        data = self.bn1(data)
        data = self.tconv(data, hextree, depth)
        data = self.bn2(data)
        return data