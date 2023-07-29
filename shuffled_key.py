# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Octree written by Peng-Shuai Wang
# Hextree modified by Xiang Wang
# --------------------------------------------------------

import torch
from typing import Optional, Union


class KeyLUT:

    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r128 = torch.arange(128, dtype=torch.int64)
        zero = torch.zeros(128, dtype=torch.int64)
        device = torch.device('cpu')

        self._encode = {device: (self.txyz2key(r128, zero, zero, zero, 7),
                                 self.txyz2key(zero, r128, zero, zero, 7),
                                 self.txyz2key(zero, zero, r128, zero, 7),
                                 self.txyz2key(zero, zero, zero, r128, 7))}
        self._decode = {device: self.key2txyz(r256, 8)}

    def encode_lut(self, device=torch.device('cpu')):
        if device not in self._encode:
            cpu = torch.device('cpu')
            self._encode[device] = tuple(e.to(device)
                                         for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device('cpu')):
        if device not in self._decode:
            cpu = torch.device('cpu')
            self._decode[device] = tuple(e.to(device)
                                         for e in self._decode[cpu])
        return self._decode[device]

    def txyz2key(self, t, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (key | ((t & mask) << (3 * i + 3)) |
                   ((x & mask) << (3 * i + 2)) |
                   ((y & mask) << (3 * i + 1)) |
                   ((z & mask) << (3 * i + 0)))
        return key

    def key2txyz(self, key, depth):
        t = torch.zeros_like(key)
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            t = t | ((key & (1 << (4 * i + 3))) >> (3 * i + 3))
            x = x | ((key & (1 << (4 * i + 2))) >> (3 * i + 2))
            y = y | ((key & (1 << (4 * i + 1))) >> (3 * i + 1))
            z = z | ((key & (1 << (4 * i + 0))) >> (3 * i + 0))
        return t, x, y, z


_key_lut = KeyLUT()


def txyz2key(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
             b: Optional[Union[torch.Tensor, int]] = None, depth: int = 14):
    r'''Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      t (torch.Tensor): The t coordinate.
      x (torch.Tensor): The x coordinate.
      y (torch.Tensor): The y coordinate.
      z (torch.Tensor): The z coordinate.
      b (torch.Tensor or int): The batch index of the coordinates, and should be 
          smaller than 128. If :attr:`b` is :obj:`torch.Tensor`, the size of
          :attr:`b` must be the same as :attr:`x`, :attr:`y`, and :attr:`z`.
      depth (int): The depth of the shuffled key, and must be smaller than 15 (< 15).
    '''
    assert (b < 128).all()
    assert depth < 15

    ET, EX, EY, EZ = _key_lut.encode_lut(x.device)
    t, x, y, z = t.long(), x.long(), y.long(), z.long()

    mask = 127 if depth > 7 else (1 << depth) - 1
    key = ET[t & mask] | EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 7:
        mask = (1 << (depth-7)) - 1
        key16 = ET[(t >> 7) & mask] | EX[(x >> 7) & mask] | EY[(
            y >> 7) & mask] | EZ[(z >> 7) & mask]
        key = key16 << 28 | key

    if b is not None:
        b = b.long()
        key = b << 56 | key

    return key


def key2txyz(key: torch.Tensor, depth: int = 14):
    r'''Decodes the shuffled key to :attr:`t`, attr:`x`, :attr:`y`, :attr:`z` coordinates
    and the batch index based on pre-computed look up tables.

    Args:
      key (torch.Tensor): The shuffled key.
      depth (int): The depth of the shuffled key, and must be smaller than 15 (< 15).
    '''
    assert depth < 15

    DT, DX, DY, DZ = _key_lut.decode_lut(key.device)
    t, x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(
        key), torch.zeros_like(key)

    b = key >> 56
    key = key & ((1 << 56) - 1)

    n = (depth + 1) // 2
    for i in range(n):
        k = key >> (i * 8) & 255
        t = t | (DT[k] << (i * 2))
        x = x | (DX[k] << (i * 2))
        y = y | (DY[k] << (i * 2))
        z = z | (DZ[k] << (i * 2))

    return t, x, y, z, b