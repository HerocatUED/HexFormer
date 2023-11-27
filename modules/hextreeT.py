# HextreeT is used to construct intermediate variables

import torch
from typing import Optional, List
from hextree import Hextree, key2txyz, key2masked


class HextreeT(Hextree):

    def __init__(self, hextree: Hextree, patch_size: int = 24, dilation: int = 4,
                 nempty: bool = True, max_depth: Optional[int] = None,
                 start_depth: Optional[int] = None, **kwargs):
        super().__init__(hextree.depth, hextree.full_depth)
        self.__dict__.update(hextree.__dict__)

        self.patch_size = patch_size
        self.dilation = dilation  # TODO dilation as a list
        self.nempty = nempty
        self.max_depth = max_depth or self.depth
        self.start_depth = start_depth or self.full_depth
        self.invalid_mask_value = -1e3
        assert self.start_depth > 1

        self.block_num = patch_size * dilation
        num = self.max_depth + 1
        # self.nnum_a = ((self.nnum_t / self.block_num).ceil()
        #                * self.block_num).int()

        self.nnum_t = [self.nnum_nempty if nempty else self.nnum] * num
        self.masked_key = [None] * num
        self.batch_idx = [None] * num
        self.patch_mask = [None] * num
        self.dilate_mask = [None] * num
        self.rel_pos = [None] * num
        self.dilate_pos = [None] * num
        self.build_t()

    def build_t(self):
        self.build_masked_key()
        for d in range(self.start_depth, self.max_depth + 1):
            self.nnum_t[d] = self.masked_key[d].shape[0]
            self.build_rel_pos(d)
            self.build_batch_idx(d)
            self.build_attn_mask(d)
            
        

    def build_batch_idx(self, depth: int):
        batch = self.batch_id_masked(depth, self.nempty)
        self.batch_idx[depth] = self.patch_partition(
            batch, depth, self.batch_size)

    def build_attn_mask(self, depth: int):
        batch = self.batch_idx[depth]
        mask = batch.view(-1, self.patch_size)
        self.patch_mask[depth] = self._calc_attn_mask(mask)

        mask = batch.view(-1, self.patch_size, self.dilation)
        mask = mask.transpose(1, 2).reshape(-1, self.patch_size)
        self.dilate_mask[depth] = self._calc_attn_mask(mask)

    def _calc_attn_mask(self, mask: torch.Tensor):
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, self.invalid_mask_value)
        return attn_mask

    def build_rel_pos(self, depth: int):
        key = self.masked_key[depth]
        key = self.patch_partition(key, depth)
        t, x, y, z, _ = key2txyz(key, depth) 
        txyz = torch.stack([t, x, y, z], dim=1)  

        txyz = txyz.view(-1, self.patch_size, 4)
        self.rel_pos[depth] = txyz.unsqueeze(2) - txyz.unsqueeze(1)

        txyz = txyz.view(-1, self.patch_size, self.dilation, 4)
        txyz = txyz.transpose(1, 2).reshape(-1, self.patch_size, 4)
        self.dilate_pos[depth] = txyz.unsqueeze(2) - txyz.unsqueeze(1)

    def patch_partition(self, data: torch.Tensor, depth: int, fill_value=0):
        assert data.shape[0] == self.nnum_t[depth], 'The shape of input data is wrong.'
        num = self.block_num - self.nnum_t[depth] % self.block_num
        tail = data.new_full((num,) + data.shape[1:], fill_value)
        return torch.cat([data, tail], dim=0)

    def patch_reverse(self, data: torch.Tensor, depth: int):
        return data[:self.nnum_t[depth]]
    
    
    def batch_id_masked(self, depth: int, nempty: bool):
        key = self.masked_key[depth]
        batch_masked = key >> 56
        return batch_masked
    
    def build_masked_key(self):
        key_masked = self.key(-1, nempty=True)
        for d in range(self.max_depth, self.start_depth-1, -1):
            key_masked = key2masked(key_masked, steps=self.depth-d)
            key_masked, _, _ = torch.unique(
                key_masked, sorted=True, return_inverse=True, return_counts=True, dim=0
            )
            self.masked_key[d] = key_masked.clone()
        