# HexFormer for Segmentation Task


import torch

from hextree import Hextree
from typing import Optional, List, Dict
from modules import HextreeInterp
from hexformer import HexFormer


class SegHeader(torch.nn.Module):

    def __init__(
            self, out_channels: int, fpn_channel: int,
            nempty: bool, num_up: int, dropout: List[float]):
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
            channels: List[int], num_blocks: List[int],
            num_heads: List[int],
            patch_size: int, dilation: int, drop_path: float,
            nempty: bool, stem_down: int, fpn_channel: int,
            head_drop: List[float], init_depth: int, **kwargs):
        super().__init__()
        self.backbone = HexFormer(
            in_channels, channels, num_blocks, num_heads, fpn_channel, patch_size, 
            dilation, drop_path, nempty, stem_down, init_depth)
        self.head = SegHeader(out_channels, fpn_channel, nempty, stem_down, head_drop)
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
