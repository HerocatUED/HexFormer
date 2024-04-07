# HexFormer for Segmentation Task


import torch

from hextree import Hextree
from typing import Optional, List, Dict
from modules import HextreeInterp
from hexformer import HexFormer


class SegHeader(torch.nn.Module):

    def __init__(
            self, out_channels: int, fpn_channel: int,
            nempty: bool, num_up: int = 2, dropout: List[float] = [0.0, 0.0]):
        super().__init__()
        self.num_up = num_up

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
        out = self.interp(features, hextree, hextree.depth, query_pts)
        out = self.classifier(out)
        return out


class HexFormerSeg(torch.nn.Module):

    def __init__(
            self, in_channels: int, out_channels: int,
            channels: List[int] = [96, 192, 384, 384],
            num_blocks: List[int] = [2, 2, 18, 2],
            num_heads: List[int] = [6, 12, 24, 24],
            patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,
            nempty: bool = True, stem_down: int = 2, head_up: int = 1, fpn_channel: int = 168,
            head_drop: List[float] = [0.0, 0.0], init_depth: int = 10, **kwargs):
        super().__init__()
        assert stem_down == head_up
        self.backbone = HexFormer(
            in_channels, channels, num_blocks, num_heads, fpn_channel, patch_size, 
            dilation, drop_path, nempty, stem_down, init_depth)
        self.head = SegHeader(out_channels, fpn_channel, nempty, head_up, head_drop)
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
