# HexFormer for Segmentation Task


import torch

from hextree import Hextree
from typing import Optional, List, Dict
from modules import HextreeInterp, HextreeUpsample, HextreeConvBnRelu, HextreeDeconvBnRelu
from hexformer import HexFormer


class SegHeader(torch.nn.Module):

    def __init__(
            self, out_channels: int, channels: List[int], fpn_channel: int,
          nempty: bool, num_up: int = 1, dropout: List[float] = [0.0, 0.0]):
        super().__init__()
        self.num_up = num_up
        self.num_stages = len(channels)

        self.conv1x1 = torch.nn.ModuleList([torch.nn.Linear(
            channels[i], fpn_channel) for i in range(self.num_stages-1, -1, -1)])
        self.upsample = HextreeUpsample('nearest', nempty)
        self.conv3x3 = torch.nn.ModuleList([HextreeConvBnRelu(
            fpn_channel, fpn_channel, kernel_size=[3],
            stride=1, nempty=nempty) for i in range(self.num_stages)])
        self.up_conv = torch.nn.ModuleList([HextreeDeconvBnRelu(
            fpn_channel, fpn_channel, kernel_size=[3],
            stride=2, nempty=nempty) for i in range(self.num_up)])
        self.interp = HextreeInterp('nearest', nempty)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout[0]),
            torch.nn.Linear(fpn_channel, fpn_channel),
            torch.nn.BatchNorm1d(fpn_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout[1]),
            torch.nn.Linear(fpn_channel, out_channels),)

    def forward(self, features: Dict[int, torch.Tensor], hextree: Hextree,
                query_pts: torch.Tensor):
        depth = min(features.keys())
        depth_max = max(features.keys())
        assert self.num_stages == len(features)

        feature = self.conv1x1[0](features[depth])
        conv_out = self.conv3x3[0](feature, hextree, depth)
        out = self.upsample(conv_out, hextree, depth, depth_max)
        for i in range(1, self.num_stages):
            depth_i = depth + i
            feature = self.upsample(feature, hextree, depth_i - 1)
            feature = self.conv1x1[i](features[depth_i]) + feature
            conv_out = self.conv3x3[i](feature, hextree, depth_i)
            out = out + self.upsample(conv_out, hextree, depth_i, depth_max)

        for i in range(self.num_up):
            out = self.up_conv[i](out, hextree, depth_max + i)  # upsample
        out = self.interp(out, hextree, depth_max + self.num_up, query_pts)
        out = self.classifier(out)
        return out


class HexFormerSeg(torch.nn.Module):

    def __init__(
            self, in_channels: int, out_channels: int,
            channels: List[int], num_blocks: List[int],
            num_heads: List[int],
            patch_size: int, dilation: int, drop_path: float,
            nempty: bool, stem_down: int, fpn_channel: int,
            head_drop: List[float], **kwargs):
        super().__init__()
        self.backbone = HexFormer(
        in_channels, channels, num_blocks, num_heads, patch_size, dilation,
        drop_path, nempty, stem_down)
        self.head = SegHeader(
            out_channels, channels, fpn_channel, nempty, stem_down, head_drop)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int,
                query_pts: torch.Tensor):
        features = self.backbone(data, hextree, depth)
        output = self.head(features, hextree, query_pts)
        return output