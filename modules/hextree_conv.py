import torch
import torch.utils.checkpoint
from typing import List

from hextree import Hextree
from ocnn.nn import OctreeConv, OctreeDeconv


class HextreeConv(torch.nn.Module):
    r"""Convolution, frame by frame with OctreeConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
    ):
        super().__init__()
        assert nempty == True, "nempty hardcode"
        self.conv = OctreeConv(in_channels, out_channels, kernel_size, stride, nempty)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        data = data[hextree.hex2oct_nempty[depth]]
        data = self.conv(data, hextree.octrees, depth)
        data = data[hextree.oct2hex_nempty[depth - 1]]
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
    ):
        super().__init__()
        assert nempty == True, "nempty hardcode"
        self.deconv = OctreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty
        )

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
    ):
        super().__init__()
        self.conv = HextreeConv(in_channels, out_channels, kernel_size, stride, nempty)
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
    ):
        super().__init__()
        self.conv = HextreeConv(in_channels, out_channels, kernel_size, stride, nempty)
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
    ):
        super().__init__()
        self.deconv = HextreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty
        )
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
    ):
        super().__init__()
        self.deconv = HextreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty
        )
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        out = self.deconv(data, hextree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out
