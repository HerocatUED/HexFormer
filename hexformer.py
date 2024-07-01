# HexFormer BackBone
import math
import torch
from typing import Optional, List
from torch.utils.checkpoint import checkpoint

from hextree import Hextree
from modules import (
    HextreeT, HextreeDropPath, 
    HextreeConvBn, HextreeConvBnRelu, 
    HextreeDeconvBn, HextreeDeconvBnRelu,
    HextreeGroupConv, HextreeResBlock,
    )


class MLP(torch.nn.Module):

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, activation=torch.nn.GELU,
                 drop: float = 0.0, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.fc1 = torch.nn.Linear(self.in_features, self.hidden_features)
        self.act = activation()
        self.fc2 = torch.nn.Linear(self.hidden_features, self.out_features)
        self.drop = torch.nn.Dropout(drop, inplace=True)

    def forward(self, data: torch.Tensor):
        data = self.fc1(data)
        data = self.act(data)
        data = self.drop(data)
        data = self.fc2(data)
        data = self.drop(data)
        return data


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

    def __init__(self, channels: int, group_size: int = 32, kernel_size: List[int] = [3], 
                 stride: int = 1, nempty: bool = False):
        super().__init__()
        self.conv = HextreeGroupConv(channels, group_size, kernel_size, stride, nempty, use_t=False)
        self.bn = torch.nn.BatchNorm1d(channels)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        data = self.conv(data, hextree, depth)
        data = self.bn(data)
        return data
    

# class PositionalEncoding(torch.nn.Module):
#     r'''sin-cos positional encoding'''
#     def __init__(self, dim, patch_size):
#         super(PositionalEncoding, self).__init__()
#         self.dim = dim
#         pe = torch.zeros(patch_size, dim)
#         position = torch.arange(0, patch_size, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Arguments:
#             x: Tensor of shape (-1, patch_size, dim)
#         Returns:
#             Tensor of shape (-1, patch_size, dim) with added positional encodings
#         """
#         x = x + self.pe.data.unsqueeze(0)
#         return x
    

class HextreeAttention(torch.nn.Module):

    def __init__(self, dim: int, patch_size: int, num_heads: int,
                 qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 dilation: int = 1, use_rpe: bool = True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.use_rpe = use_rpe
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads, dilation) if self.use_rpe else None
        # self.pe = PositionalEncoding(dim, patch_size)

    def forward(self, data: torch.Tensor, hextree: HextreeT, depth: int):
        H = self.num_heads
        K = self.patch_size
        C = self.dim
        D = self.dilation

        # patch partition
        data = hextree.patch_partition(data, depth)
        if D > 1:  # dilation
            rel_pos = hextree.dilate_pos[depth]
            mask = hextree.dilate_mask[depth]
            data = data.view(-1, K, D, C).transpose(1, 2).reshape(-1, C)
        else:
            rel_pos = hextree.rel_pos[depth]
            mask = hextree.patch_mask[depth]
        data = data.view(-1, K, C)
        # data = self.pe(data)

        # qkv
        qkv = self.qkv(data).reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # (N, H, K, C')
        q = q * self.scale

        # attn
        attn = q @ k.transpose(-2, -1)        # (N, H, K, K)
        attn = self.apply_rpe(attn, rel_pos)  # (N, H, K, K)
        attn = attn + mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        data = (attn @ v).transpose(1, 2).reshape(-1, C)

        # patch reverse
        if D > 1:  # dilation
            data = data.view(-1, D, K, C).transpose(1, 2).reshape(-1, C)
        data = hextree.patch_reverse(data, depth)

        # ffn
        data = self.proj(data)
        data = self.proj_drop(data)
        return data

    def apply_rpe(self, attn, rel_pos):
        if self.use_rpe:
            if isinstance(self.rpe, RPE2):
                w, b = self.rpe(rel_pos)
                attn = w*attn + b
            elif isinstance(self.rpe, RPE):
                attn = attn + self.rpe(rel_pos) 
            else: 
                assert NotImplementedError, 'only RPE and RPE2 implemented!'
        return attn

    def extra_repr(self) -> str:
        return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
                self.dim, self.patch_size, self.num_heads, self.dilation)  # noqa


class HexFormerBlock(torch.nn.Module):

    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                 dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, **kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attention = HextreeAttention(dim, patch_size, num_heads, qkv_bias,
                                          qk_scale, attn_drop, proj_drop, dilation)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
        self.drop_path = HextreeDropPath(drop_path, nempty)
        self.cpe = CPE(dim, nempty=nempty)

    def forward(self, data: torch.Tensor, hextree: HextreeT, depth: int):
        data = self.cpe(data, hextree, depth) + data
        attn = self.attention(self.norm1(data), hextree, depth)
        data = data + self.drop_path(attn, hextree, depth)
        ffn = self.mlp(self.norm2(data))
        data = data + self.drop_path(ffn, hextree, depth)
        return data


class HexFormerStage(torch.nn.Module):

    def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
                 dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: Optional[float] = None, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
                 use_checkpoint: bool = True, num_blocks: int = 2,
                 hexformer_block=HexFormerBlock, **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        self.use_checkpoint = use_checkpoint

        self.blocks = torch.nn.ModuleList([hexformer_block(dim=dim, num_heads=num_heads, patch_size=patch_size,
                                                           dilation=1 if (i % 2 == 0) else dilation,
                                                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop,
                                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                           nempty=nempty, activation=activation) for i in range(num_blocks)])

    def forward(self, data: torch.Tensor, hextree: HextreeT, depth: int):
        for i in range(self.num_blocks):
            if self.use_checkpoint and self.training:
                data = checkpoint(self.blocks[i], data, hextree, depth, use_reentrant=False)
            else:
                data = self.blocks[i](data, hextree, depth)
        return data
    
    
class PatchEmbed(torch.nn.Module):

    def __init__(self, in_dim: int, dim: int, num_down: int, nempty: bool, **kwargs):
        super().__init__()
        self.num_stages = num_down
        channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

        self.convs = torch.nn.ModuleList([HextreeConvBnRelu(
            in_dim if i == 0 else channels[i], channels[i], kernel_size=[3],
            stride=1, nempty=nempty) for i in range(self.num_stages)])
        # self.convs = torch.nn.ModuleList([HextreeResBlock(
        #     in_dim if i == 0 else channels[i], channels[i],
        #     stride=1, nempty=nempty, use_t=False) for i in range(self.num_stages)])
        self.downsamples = torch.nn.ModuleList([HextreeConvBnRelu(
            channels[i], channels[i+1], kernel_size=[2], stride=2, nempty=nempty)
            for i in range(self.num_stages)])
        self.proj = HextreeConvBnRelu(
            channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.convs[i](data, hextree, depth_i)
            data = self.downsamples[i](data, hextree, depth_i)
        data = self.proj(data, hextree, depth_i - 1)
        return data
    

class HexFormer(torch.nn.Module):

    def __init__(self, in_channels: int, channels: List[int],
                 num_blocks: List[int], num_heads: List[int],
                 patch_size: int, dilation: int, drop_path: float, 
                 nempty: bool, stem_down: int, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.dilation = dilation
        self.nempty = nempty
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        encode_blocks = sum(num_blocks)
        drop_ratio = torch.linspace(0, drop_path, encode_blocks).tolist()
        # drop_ratio = torch.linspace(0, drop_path, encode_blocks+sum(num_blocks[:-1])).tolist()
        # Patch Embdedding
        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        # Encoder
        self.encoders = torch.nn.ModuleList([HexFormerStage(
            dim=channels[i], num_heads=num_heads[i], patch_size=patch_size,
            drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i+1])],
            dilation=dilation, nempty=nempty, num_blocks=num_blocks[i],)
            for i in range(self.num_stages)])
        self.downsamples = torch.nn.ModuleList([HextreeConvBn(
        channels[i], channels[i + 1], kernel_size=[2], stride=2, 
        nempty=nempty, use_bias=True) for i in range(self.num_stages - 1)])
        # Decoder
        # self.decoders = torch.nn.ModuleList([HexFormerStage(
        #     dim=channels[-i-2], num_heads=num_heads[-i-2], patch_size=patch_size,
        #     drop_path=drop_ratio[encode_blocks+sum(num_blocks[-i-1:-1]):encode_blocks+sum(num_blocks[-i-2:-1])],
        #     dilation=dilation, nempty=nempty, num_blocks=num_blocks[-i-2],)
        #     for i in range(self.num_stages - 1)])
        # self.upsamples = torch.nn.ModuleList([HextreeDeconvBn(
        # channels[-i-1], channels[-i-2], kernel_size=[2], stride=2, 
        # nempty=nempty, use_bias=True) for i in range(self.num_stages - 1)])
        
    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        # PE
        data = self.patch_embed(data, hextree, depth)
        depth = depth - self.stem_down   # current hextree depth
        hextree = HextreeT(hextree, self.patch_size, self.dilation, self.nempty,
                           max_depth=depth, start_depth=depth-self.num_stages+1)
        features = {}
        # Encoder
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.encoders[i](data, hextree, depth_i)
            features[depth_i] = data
            if i < self.num_stages - 1:
                data = self.downsamples[i](data, hextree, depth_i)
        # Decoder
        # for i in range(self.num_stages - 1):
        #     depth_i = depth - self.num_stages + i + 1
        #     data = self.upsamples[i](data, hextree, depth_i)
        #     data = self.decoders[i](data, hextree, depth_i + 1)
        #     features[depth_i + 1] += data
        return features