# HexFormer for Segmentation Task


import torch

from hextree import Hextree
from typing import Optional, List, Dict
from hexformer import HexFormer
import ocnn

class ActSegHeader(torch.nn.Module):

    def __init__(
            self, out_channel: int, in_channel: int, hid_channel: int,
            nempty: bool, dropout: float):
        super().__init__()
        self.in_channel = in_channel
        self.global_pool = ocnn.nn.OctreeGlobalPool(nempty)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channel, hid_channel),
            torch.nn.BatchNorm1d(hid_channel),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(p = dropout),
            torch.nn.Linear(hid_channel, out_channel))

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        data = data[hextree.hex2oct_nempty[depth]]
        data = self.global_pool(data, hextree.octrees, depth)
        # NOTE: there is no need to do inverse indexing
        logit = self.classifier(data)
        return logit # (B * 150, 19)


class HexFormerActSeg(torch.nn.Module):

    def __init__(
            self, in_channels: int, out_channels: int,
            channels: List[int], num_blocks: List[int],
            num_heads: List[int],
            patch_size: int, dilation: int, drop_path: float,
            nempty: bool, stem_down: int, hid_channel: int, 
            head_drop: float, **kwargs):
        super().__init__()
        self.backbone = HexFormer(
        in_channels, channels, num_blocks, num_heads, patch_size, dilation,
        drop_path, nempty, stem_down)
        self.head = ActSegHeader(
            out_channels, channels[-1], hid_channel, nempty, head_drop)
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