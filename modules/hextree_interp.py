import torch
import torch.sparse
from typing import List, Optional
from ocnn.nn import OctreeInterp, OctreeUpsample
from hextree import Hextree


def hextree_nearest_pts(
    data: torch.Tensor,
    hextree: Hextree,
    depth: int,
    pts: torch.Tensor,
    nempty: bool = False,
    bound_check: bool = False,
):
    """The nearest-neighbor interpolatation with input points.

    Args:
      data (torch.Tensor): The input data.
      hextree (Hextree): The hextree to interpolate.
      depth (int): The depth of the data.
      pts (torch.Tensor): The coordinates of the points with shape :obj:`(N, 5)`,
          i.e. :obj:`N x (t, x, y, z, batch)`.
      nempty (bool): If true, the :attr:`data` only contains features of non-empty
          hextree nodes
      bound_check (bool): If true, check whether the point is in :obj:`[0, 2^depth)`.

    .. note::
      The :attr:`pts` MUST be scaled into :obj:`[0, 2^depth)`.
    """

    nnum = hextree.nnum_nempty[depth] if nempty else hextree.nnum[depth]
    assert data.shape[0] == nnum, "The shape of input data is wrong."

    idx = hextree.search_txyzb(pts, depth, nempty)
    valid = idx > -1  # valid indices
    if bound_check:
        bound = torch.logical_and(pts[:, :4] >= 0, pts[:, :4] < 2**depth).all(1)
        valid = torch.logical_and(valid, bound)

    size = (pts.shape[0], data.shape[1])
    out = torch.zeros(size, device=data.device, dtype=data.dtype)
    out[valid] = data.index_select(0, idx[valid])
    return out


class HextreeInterp(torch.nn.Module):
    r"""Interpolates the points with an hextree feature.

    Refer to :func:`hextree_nearest_pts` for a description of arguments.
    """

    def __init__(
        self,
        method: str = "linear",
        nempty: bool = False,
        bound_check: bool = False,
        rescale_pts: bool = True,
    ):
        super().__init__()
        assert method == "nearest"
        self.method = method
        self.nempty = nempty
        self.bound_check = bound_check
        self.rescale_pts = rescale_pts
        self.func = hextree_nearest_pts

    def forward(
        self, data: torch.Tensor, hextree: Hextree, depth: int, pts: torch.Tensor
    ):
        r""""""

        # rescale points from [-1, 1] to [0, 2^depth], NOTE: do not rescale t!!!
        if self.rescale_pts:
            scale = 2 ** (depth - 1)
            pts[:, 1:4] = (pts[:, 1:4] + 1.0) * scale

        return self.func(data, hextree, depth, pts, self.nempty, self.bound_check)

    def extra_repr(self) -> str:
        r"""Sets the extra representation of the module."""

        return ("method={}, nempty={}, bound_check={}, rescale_pts={}").format(
            self.method, self.nempty, self.bound_check, self.rescale_pts
        )  # noqa


# class HextreeInterp(torch.nn.Module):
#     r"""Interpolates the points with an hextree feature.

#     Refer to :func:`hextree_nearest_pts` for a description of arguments.
#     """

#     def __init__(
#         self,
#         method: str = "linear",
#         nempty: bool = False,
#         bound_check: bool = False,
#         rescale_pts: bool = True,
#     ):
#         super().__init__()
#         self.interp = OctreeInterp(method, nempty, bound_check, rescale_pts)

#     def forward(
#         self, data: torch.Tensor, hextree: Hextree, depth: int, pts: torch.Tensor
#     ):
#         data = data[hextree.hex2oct_nempty[depth]]
#         data = self.interp(data, hextree.octrees, depth, pts)
#         data = data[hextree.oct2hex_nempty[depth]]
#         return data


class HextreeUpsample(torch.nn.Module):
    r"""Upsamples the hextree node features from :attr:`depth` to
    :attr:`(target_depth)`.

    Refer to :class:`hextree_nearest_pts` for details.
    """

    def __init__(self, method: str = "linear", nempty: bool = False):
        super().__init__()
        self.upsample = OctreeUpsample(method, nempty)

    def forward(
        self,
        data: torch.Tensor,
        hextree: Hextree,
        depth: int,
        target_depth: Optional[int] = None,
    ):
        if target_depth is None:
            target_depth = depth + 1
        data = data[hextree.hex2oct_nempty[depth]]
        data = self.upsample(data, hextree.octrees, depth, target_depth)
        data = data[hextree.oct2hex_nempty[target_depth]]
        return data
