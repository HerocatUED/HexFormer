# HexFormer BackBone

import torch
import ocnn
from typing import Optional, List

from hextree import Hextree


class RPE(torch.nn.Module):

    def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.pos_bnd = self.get_pos_bnd(patch_size)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(4*self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def get_pos_bnd(self, patch_size: int):
        return int(0.8 * patch_size * self.dilation**0.5)

    def txyz2idx(self, txyz: torch.Tensor):
        mul = torch.arange(4, device=txyz.device) * self.rpe_num
        txyz = txyz.clamp(-self.pos_bnd, self.pos_bnd)
        idx = txyz + (self.pos_bnd + mul)
        return idx

    def forward(self, txyz):
        idx = self.txyz2idx(txyz)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

    def extra_repr(self) -> str:
        return 'num_heads={}, pos_bnd={}, dilation={}'.format(
                self.num_heads, self.pos_bnd, self.dilation)  # noqa
        

class RPE2(torch.nn.Module):

    def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.pos_bnd = self.get_pos_bnd(patch_size)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table_w = torch.nn.Parameter(torch.zeros(4*self.rpe_num, num_heads))
        self.rpe_table_b = torch.nn.Parameter(torch.zeros(4*self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table_w, std=0.02)
        torch.nn.init.trunc_normal_(self.rpe_table_b, std=0.02)

    def get_pos_bnd(self, patch_size: int):
        return int(0.8 * patch_size * self.dilation**0.5)

    def txyz2idx(self, txyz: torch.Tensor):
        mul = torch.arange(4, device=txyz.device) * self.rpe_num
        txyz = txyz.clamp(-self.pos_bnd, self.pos_bnd)
        idx = txyz + (self.pos_bnd + mul)
        return idx

    def forward(self, txyz):
        idx = self.txyz2idx(txyz)
        w = self.rpe_table_w.index_select(0, idx.reshape(-1))
        w = w.view(idx.shape + (-1,)).sum(3)
        w = w.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        b = self.rpe_table_b.index_select(0, idx.reshape(-1))
        b = b.view(idx.shape + (-1,)).sum(3)
        b = b.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return w, b

    def extra_repr(self) -> str:
        return 'num_heads={}, pos_bnd={}, dilation={}'.format(
                self.num_heads, self.pos_bnd, self.dilation)  # noqa
        
        
class CPE(torch.nn.Module):

    def __init__(self, in_channels: int, kernel_size: List[int] = [3],
                 group_size: int = 32, stride: int = 1, nempty: bool = False):
        super().__init__()
        assert in_channels % group_size == 0
        self.group_size = group_size
        self.group_num = in_channels // group_size
        self.convs = [ocnn.nn.OctreeConv(
            group_size, group_size, kernel_size, 
            stride, nempty, use_bias=True) for i in range(self.group_num)]
        self.bn = torch.nn.BatchNorm1d(in_channels)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        # data (N, C), C = k * group_size
        assert data.size(dim = 1) % self.group_size == 0
        data = data[hextree.hex2oct_nempty[depth]]
        out = torch.zeros_like(data)
        for i in range(self.group_num):
            out[:, self.group_num*i:self.group_size*(i+1)] = self.convs[i](data[:, self.group_num*i:self.group_size*(i+1)], hextree.octrees, depth)
        out = self.bn(out)
        data = data[hextree.oct2hex_nempty[depth]]
        return out