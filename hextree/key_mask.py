import torch


class KeyMaskLUT:
    def __init__(self):
        device = torch.device('cpu')
        self._encode = {device: torch.tensor([
            0x7fffffffffffffff, 0x7ffffffffffffff8, 0x7fffffffffffff88, 0x7ffffffffffff888,
            0x7fffffffffff8888, 0x7ffffffffff88888, 0x7fffffffff888888, 0x7ffffffff8888888,
            0x7fffffff88888888, 0x7ffffff888888888, 0x7fffff8888888888, 0x7ffff88888888888,
            0x7fff888888888888, 0x7ff8888888888888, 0x7f88888888888888
        ], dtype=torch.int64)}

    def encode_lut(self, device=torch.device('cpu')):
        if device not in self._encode:
            cpu = torch.device('cpu')
            self._encode[device] = tuple(e.to(device)
                                         for e in self._encode[cpu])
        return self._encode[device]


_scatter_mask_lut = KeyMaskLUT()


def key2masked(key: torch.Tensor, steps: int, need_index: bool=False):
    keys_masked = key.long()
    mask_lut = _scatter_mask_lut.encode_lut(key.device)
    xyz_index = keys_masked & ~mask_lut[steps]
    keys_masked = keys_masked & mask_lut[steps]
    if need_index:
        return keys_masked, xyz_index
    return keys_masked