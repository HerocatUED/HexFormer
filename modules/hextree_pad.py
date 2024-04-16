import torch
from hextree import Hextree


def hextree_pad(data: torch.Tensor, hextree: Hextree, depth: int, val: float = 0.0):
    r"""Pads :attr:`val` to make the number of elements of :attr:`data` equal to
    the hextree node number.

    Args:
      data (torch.Tensor): The input tensor with its number of elements equal to the
          non-empty hextree node number.
      hextree (Hextree): The corresponding hextree.
      depth (int): The depth of current hextree.
      val (float): The padding value. (Default: :obj:`0.0`)
    """

    mask = hextree.nempty_mask(depth)
    size = (hextree.nnum[depth], data.shape[1])  # (N, C)
    out = torch.full(size, val, dtype=data.dtype, device=data.device)
    out[mask] = data
    return out


def hextree_depad(data: torch.Tensor, hextree: Hextree, depth: int):
    r"""Reverse operation of :func:`hextree_depad`.

    Please refer to :func:`hextree_depad` for the meaning of the arguments.
    """

    mask = hextree.nempty_mask(depth)
    return data[mask]
