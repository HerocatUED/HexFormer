import torch
import torch.utils.checkpoint
from typing import List

from ocnn.nn import OctreeConv, OctreeDeconv
from ocnn.octree import Octree
from ocnn.modules import OctreeConvBnRelu, OctreeDeconvBnRelu, OctreeConvBn


class OctreeConvBn(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv` and :obj:`BatchNorm`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.bn(out)
    return out


class OctreeConvBnRelu(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv`, :obj:`BatchNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.bn(out)
    out = self.relu(out)
    return out


class OctreeDeconvBnRelu(torch.nn.Module):
  r''' A sequence of :class:`OctreeDeconv`, :obj:`BatchNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeDeconv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.deconv = OctreeDeconv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.deconv(data, octree, depth)
    out = self.bn(out)
    out = self.relu(out)
    return out
