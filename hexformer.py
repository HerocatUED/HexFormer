# HexFormer BackBone

import torch
from typing import Optional, List
from torch.utils.checkpoint import checkpoint

from hextree import Hextree
from modules import HextreeDropPath, HextreeAvgPoolXYZ, HextreeT


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
            attn = attn + self.rpe(rel_pos)
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
        # self.cpe = OctreeDWConvBn(dim, nempty=nempty)

    def forward(self, data: torch.Tensor, hextree: HextreeT, depth: int):
        # data = self.cpe(data, hextree, depth) + data
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
        self.interval = interval  # normalization interval
        self.num_norms = (num_blocks - 1) // self.interval

        self.blocks = torch.nn.ModuleList([hexformer_block(dim=dim, num_heads=num_heads, patch_size=patch_size,
                                                           dilation=1 if (i % 2 == 0) else dilation,
                                                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop,
                                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                           nempty=nempty, activation=activation) for i in range(num_blocks)])
        # self.norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])

    def forward(self, data: torch.Tensor, hextree: HextreeT, depth: int):
        for i in range(self.num_blocks):
            if self.use_checkpoint and self.training:
                data = checkpoint(self.blocks[i], data, hextree, depth)
            else:
                data = self.blocks[i](data, hextree, depth)
            # if i % self.interval == 0 and i != 0:
            #   data = self.norms[(i - 1) // self.interval](data)
        return data


class PatchEmbed(torch.nn.Module):

    def __init__(self, in_dim: int = 4, dim: int = 96, num_stages: int = 2, nempty: bool = True, **kwargs):
        super().__init__()
        self.num_stages = num_stages
        channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

        self.mlps = torch.nn.ModuleList(
            [MLP(in_dim if i == 0 else channels[i-1], channels[i], channels[i]) for i in range(self.num_stages)])
        self.norm = torch.nn.LayerNorm(channels[-2])
        self.downsample = HextreeAvgPoolXYZ()
        self.proj = MLP(channels[-2], 2*dim, channels[-1])

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.mlps[i](data)
            data = self.downsample(data, hextree, depth_i)
            
        data = self.proj(self.norm(data))
        return data


class HexFormer(torch.nn.Module):

    def __init__(self, in_channels: int,
                 channels: List[int] = [96, 192, 384, 384],
                 num_blocks: List[int] = [2, 2, 18, 2],
                 num_heads: List[int] = [6, 12, 24, 24],
                 patch_size: int = 32, dilation: int = 4, drop_path: float = 0.5,
                 nempty: bool = True, stem_down: int = 2, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.dilation = dilation
        self.nempty = nempty
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(
            in_channels, channels[0], stem_down, nempty)
        self.layers = torch.nn.ModuleList([HexFormerStage(
            dim=channels[i], num_heads=num_heads[i], patch_size=patch_size,
            drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i+1])],
            dilation=dilation, nempty=nempty, num_blocks=num_blocks[i],)
            for i in range(self.num_stages)])
        self.feature_up = torch.nn.ModuleList([MLP(channels[i], int(
            (channels[i]+channels[i+1])/2), channels[i+1]) for i in range(self.num_stages - 2)])
        self.downsample = HextreeAvgPoolXYZ()

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        data = self.patch_embed(data, hextree, depth)
        depth = depth - self.stem_down   # current hextree depth
        hextree = HextreeT(hextree, self.patch_size, self.dilation, self.nempty,
                           max_depth=depth, start_depth=depth-self.num_stages+1)
        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.layers[i](data, hextree, depth_i)
            features[depth_i] = data
            if i < self.num_stages - 1:
                data = self.downsample(data, hextree, depth_i)
                if i < self.num_stages - 2:
                    data = self.feature_up[i](data)
        return features
