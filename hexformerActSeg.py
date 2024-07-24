# HexFormer for Segmentation Task


import torch

from hextree import Hextree, key2txyz
from typing import Optional, List, Dict
from modules import HextreeConvBnRelu, HextreeDeconvBnRelu
from hexformer import HexFormer


class ActSegHeader(torch.nn.Module):

    def __init__(
            self, out_channel: int, in_channel: int, hid_channel: int,
          head_down: int, nempty: bool, dropout: float):
        super().__init__()
        self.head_down = head_down
        self.in_channel = in_channel
        # self.conv3x3 = torch.nn.ModuleList([HextreeConvBnRelu(
        #     in_channel, in_channel, kernel_size=[3],
        #     stride=1, nempty=nempty) for i in range(self.head_down)])
        self.downsamples = torch.nn.ModuleList([HextreeConvBnRelu(
            in_channel, in_channel, kernel_size=[2], stride=2, nempty=nempty)
            for i in range(head_down)])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channel * 8, hid_channel),
            torch.nn.BatchNorm1d(hid_channel),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p = dropout),
            torch.nn.Linear(hid_channel, out_channel))

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        # downsample to depth = 1
        for i in range(self.head_down):
            # data = self.conv3x3[i](data, hextree, depth)
            data = self.downsamples[i](data, hextree, depth)
            depth -= 1
        assert depth == 1
        # get current feature
        t, x, y, z, b = key2txyz(hextree.key(depth, True))
        datas = []
        for i in torch.unique(b):
            mask0 = b == i
            t_ = t[mask0]
            mask1 = t_ == t_.max()
            datas.append(data[mask0][mask1].view(1, -1))
        out = torch.vstack(datas)
        # classify
        logit = self.classifier(out)
        return logit


class HexFormerActSeg(torch.nn.Module):

    def __init__(
            self, in_channels: int, out_channels: int,
            channels: List[int], num_blocks: List[int],
            num_heads: List[int],
            patch_size: int, dilation: int, drop_path: float,
            nempty: bool, stem_down: int, head_down: int, hid_channel: int, 
            head_drop: float, **kwargs):
        super().__init__()
        self.backbone = HexFormer(
        in_channels, channels, num_blocks, num_heads, patch_size, dilation,
        drop_path, nempty, stem_down)
        self.head = ActSegHeader(
            out_channels, channels[-1], hid_channel, head_down, nempty, head_drop)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        features = self.backbone(data, hextree, depth)
        curr_depth = min(features.keys())
        output = self.head(features[curr_depth], hextree, curr_depth)
        return output