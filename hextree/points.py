import torch
import numpy as np
from typing import Optional, Union, List


class Points:
    r"""Represents a 4D pointcloud and contains some elementary transformations.

    Args:
        :obj:`points` (torch.Tensor): The txyz coordinates of the points with a shape of
            :obj:`(N, 4)`, where :obj:`N` is the number of points.
        :obj:`normals` (torch.Tensor or None): The point normals with a shape of
            :obj:`(N, 3)`, default :obj:`None`
        :obj:`features` (torch.Tensor or None): The point features with a shape of
            :obj:`(N, C)`, where :obj:`C` is the channel of features, default :obj:`None`
        :obj:`labels` (torch.Tensor or None): The point labels with a shape of
            :obj:`(N, K)`, where :obj:`K` is the channel of labels, default :obj:`None`
        :obj:`batch_id` (torch.Tensor or None): The batch indices for each point with a
            shape of :obj:`(N, 1)`, default :obj:`None`
        :obj:`batch_size` (:obj:`int`): The batch size, default 1

        Currently `batch_size` and `batch_id` are all ignored, i.e. batch_size is always viewed as 1
        when buiding `Points` from raw data, and they are automatically set after `merge_points`
    """

    def __init__(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        batch_id: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ):
        super().__init__()

        self.points = points
        self.normals = normals
        self.features = features
        self.labels = labels
        # TODO: Allow batch operation
        self.batch_id = None  # valid after `merge_points`
        self.batch_size = 1  # valid after `merge_points`
        # self.batch_id = batch_id
        # self.batch_size = batch_size

        self.device = points.device

        # TODO: Allow batch operation
        self.batch_npt = None  # valid after `merge_points`
        # self.batch_npt = None

    # Properties
    @property
    def npt(self):
        r"""Returns the number of points"""
        return self.points.shape[0]

    def bbox_xyz(self):
        r"""Returns the xyz bounding box `(bbmin, bbmax)`, each shaped `(3,)`."""

        # torch.min and torch.max return (value, indices)
        bbmin = self.points.min(dim=0)
        bbmax = self.points.max(dim=0)
        return bbmin[0][1:], bbmax[0][1:]

    def inbox_mask(
        self,
        bbmin: Union[float, torch.Tensor] = -1.0,
        bbmax: Union[float, torch.Tensor] = 1.0,
    ):
        r"""Returns a mask indicating whether the points are within the specified
        bounding box or not.
        """

        mask_min = torch.all(self.points[:, 1:] > bbmin, dim=1)
        mask_max = torch.all(self.points[:, 1:] < bbmax, dim=1)
        mask = torch.logical_and(mask_min, mask_max)
        return mask

    # Geometric operations
    def scale_xyz(self, factor: Union[torch.Tensor, float, int]):
        r"""Scale the pointcloud on xyz axis.

        Args:
            `factor` (`torch.Tensor`): The scaling factor, can be
            a tensor with shape `(3,)` or `(1,)`, or a float number
        """

        # convert all types to tensor
        if isinstance(factor, (int, float)):
            factor = torch.tensor([factor], dtype=float)

        # special case check
        non_zero = (factor != 0).all()
        assert non_zero, "The scale factor must not constain 0."
        all_ones = (factor == 1.0).all()
        if all_ones:
            return

        # point scaling
        factor = factor.to(self.device)
        self.points[:, 1:] = self.points[:, 1:] * factor

        # normal transform
        non_uniform = (factor != factor[0]).any()
        if self.normals is not None and non_uniform:
            ifactor = 1.0 / factor
            self.normals = self.normals * ifactor
            norm2 = torch.sqrt(torch.sum(self.normals**2, dim=1, keepdim=True))
            self.normals = self.normals / torch.clamp(norm2, min=1.0e-12)

    def scale_txyz(self, factor: torch.Tensor):
        r"""Scale the pointcloud on txyz, but we do not recommend scaling t.

        Function `scale_xyz` do scaling on xyz axes only and might be a better one to use.

        Args:
            factor (torch.Tensor): The scaling factor with shape `(4,)` or `(2,)`,
                if shape is `(2,)`, t will be scaled with `factor[0]` and xyz will be scaled with `factor[1]`
        """

        # dimension check
        assert len(factor) in [4, 2], "Length of scaling factor must be 2 or 4."

        # special case check
        non_zero = (factor != 0).all()
        assert non_zero, "The scale factor must not constain 0."
        all_ones = (factor == 1.0).all()
        if all_ones:
            return

        # t axis scaling
        self.points[:, 0] = self.points[:, 0] * factor[0]

        # xyz scaling and normal transform
        self.scale_xyz(factor[1:])

    def flip(self, axis: str = "x"):
        r"""Flips the pointcloud along the given axis.

        Args:
            `axis` (`str`): The flipping axis, combination of `'x'`, `'y'`, and `'z'`. Default `'x'`.
        """
        axis_map = {"x": 1, "y": 2, "z": 3}
        for ax in axis:
            idx = axis_map[ax]
            self.points[:, idx] *= -1.0
            if self.normals is not None:
                self.normals[:, idx - 1] *= -1.0

    def translate_xyz(self, dis: Union[torch.Tensor, float, int]):
        r"""Translates the pointcloud on xyz by `dis`.

        Args:
            `dis` (`torch.Tensor`): The displacement with shape `(3,)` or `(1,)`, or a float
        """

        # convert all types to tensor
        if isinstance(dis, (int, float)):
            dis = torch.tensor([dis], dtype=float)

        dis = dis.to(self.device)
        self.points[:, 1:] = self.points[:, 1:] + dis

    def translate_txyz(self, dis: torch.Tensor):
        r"""Translates the pointcloud on txyz by `dis`, but we do not recommend translating t.

        Function `translate_xyz` do scaling on xyz axes only and might be a better one to use.

        Args:
            `dis` (`torch.Tensor`): The displacement with shape `(4,)` or `(2,)`,
                if shape is `(2,)`, t will be translate by `dis[0]` and xyz will be translate with `dis[1]`
        """

        dis = dis.to(self.device)
        self.points[:, 0] = self.points[:, 0] + dis[0]
        self.translate_xyz(dis[1:])

    def normalize_xyz(self, keep_shape: bool = True):
        r"""Normalizes the pointcloud to :obj:`[-1, 1]`.

        Args:
            `keep_shape` (`bool`): if set to `False`, do scaling on each axis seperately;
            otherwise xyz will be scaled by the same factor. Default `True`.
        """

        bbmin, bbmax = self.bbox_xyz()
        center = (bbmax + bbmin) * 0.5
        self.translate_xyz(-center)
        box_size = bbmax - bbmin + 1.0e-6
        if keep_shape:
            box_size = box_size.max(0, keepdim=True)[0]
        self.scale_xyz(2.0 / box_size)

    def clip_xyz(
        self,
        bbmin: Union[float, torch.Tensor] = -1.0,
        bbmax: Union[float, torch.Tensor] = 1.0,
        eps: Union[float, torch.Tensor] = 0.01,
    ):
        r"""Clips the pointcloud to :obj:`[min+eps, max-eps]` and returns the mask.

        Args:
            bbmin (Union[float, torch.Tensor]): The minimum value to clip.
            bbmax (Union[float, torch.Tensor]): The maximum value to clip.
            eps (Union[float, torch.Tensor]): The margin.
        """
        mask = self.inbox_mask(bbmin + eps, bbmax - eps)
        tmp = self.__getitem__(mask)
        self.__dict__.update(tmp.__dict__)
        return mask

    def orient_normal(self, axis: str = "x"):
        r"""Orients the point normals along a given axis.

        Args:
            axis (str): The coordinate axes, choose from :obj:`x`, :obj:`y`, :obj:`z`
                and :obj:`xyz`. (default: :obj:`x`)
        """

        if self.normals is None:
            return
        axis_map = {"x": 0, "y": 1, "z": 2, "xyz": 3}
        idx = axis_map[axis]
        if idx < 3:
            flags = self.normals[:, idx] > 0
            flags = flags.float() * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            self.normals = self.normals * flags.unsqueeze(1)
        else:
            self.normals.abs_()

    def rotate(self, angle: torch.Tensor):
        r"""Rotates the pointcloud.

        Args:
            angle (torch.Tensor): The rotation angles in radian with shape :obj:`(3, )`.
        """

        vcos, vsin = angle.cos(), angle.sin()
        # rotx, roty, rotz are actually the transpose of the rotation matrices
        rotx = torch.Tensor([[1, 0, 0], [0, vcos[0], vsin[0]], [0, -vsin[0], vcos[0]]])
        roty = torch.Tensor([[vcos[1], 0, -vsin[1]], [0, 1, 0], [vsin[1], 0, vcos[1]]])
        rotz = torch.Tensor([[vcos[2], vsin[2], 0], [-vsin[2], vcos[2], 0], [0, 0, 1]])
        rot = rotx @ roty @ rotz

        rot = rot.to(self.device)
        self.points[:, 1:] = self.points[:, 1:] @ rot
        if self.normals is not None:
            self.normals = self.normals @ rot

    # Array operations
    def __getitem__(self, mask: torch.Tensor):
        r"""Slices the pointcloud according a given :attr:`mask`."""

        dummy_pts = torch.zeros(1, 4, device=self.device)
        out = Points(dummy_pts, batch_size=self.batch_size)

        out.points = self.points[mask]
        if self.normals is not None:
            out.normals = self.normals[mask]
        if self.features is not None:
            out.features = self.features[mask]
        if self.labels is not None:
            out.labels = self.labels[mask]
        if self.batch_id is not None:
            out.batch_id = self.batch_id[mask]
        return out

    # Device operations
    def to(self, device: Union[torch.device, str], non_blocking: bool = False):
        r"""Moves the Points to a specified device.

        Args:
        device (torch.device or str): The destination device.
        non_blocking (bool): If True and the source is in pinned memory, the copy
            will be asynchronous with repsect to the host. Otherwise, the argument
            has no effect. Default: False.
        """

        if isinstance(device, str):
            device = torch.device(device)

        #  If on the save device, directly retrun self
        if self.device == device:
            return self

        # Construct a new Points on the specified device
        points = Points(torch.zeros(1, 3, device=device))
        points.batch_npt = self.batch_npt
        points.points = self.points.to(device, non_blocking=non_blocking)
        if self.normals is not None:
            points.normals = self.normals.to(device, non_blocking=non_blocking)
        if self.features is not None:
            points.features = self.features.to(device, non_blocking=non_blocking)
        if self.labels is not None:
            points.labels = self.labels.to(device, non_blocking=non_blocking)
        if self.batch_id is not None:
            points.batch_id = self.batch_id.to(device, non_blocking=non_blocking)
        return points

    def cuda(self, non_blocking: bool = False):
        r"""Moves the Points to the GPU."""

        return self.to("cuda", non_blocking)

    def cpu(self):
        r"""Moves the Points to the CPU."""

        return self.to("cpu")

    def save(self, filename: str, info: str = "PNFL"):
        r"""Save the Points into npz or xyz files.

        Args:
        filename (str): The output filename.
        info (str): The infomation for saving: 'P' -> 'points', 'N' -> 'normals',
            'F' -> 'features', 'L' -> 'labels', 'B' -> 'batch_id'.
        """

        mapping = {
            "P": ("points", self.points),
            "N": ("normals", self.normals),
            "F": ("features", self.features),
            "L": ("labels", self.labels),
            "B": ("batch_id", self.batch_id),
        }

        names, outs = [], []
        for key in info.upper():
            name, out = mapping[key]
            if out is not None:
                names.append(name)
                if out.dim() == 1:
                    out = out.unsqueeze(1)
                outs.append(out.cpu().numpy())

        if filename.endswith("npz"):
            out_dict = dict(zip(names, outs))
            np.savez(filename, **out_dict)
        elif filename.endswith("xyz"):
            out_array = np.concatenate(outs, axis=1)
            np.savetxt(filename, out_array, fmt="%.6f")
        else:
            raise ValueError


def merge_points(points: List["Points"], update_batch_info: bool = True):
    r"""Merges a list of points

    Args:
        `points` (`List[Points]`): A list of points to merge.

    Currently information related to batch are all ignored.
    """

    out = Points(torch.zeros(1, 4))
    out.points = torch.cat([p.points for p in points], dim=0)
    if points[0].normals is not None:
        out.normals = torch.cat([p.normals for p in points], dim=0)
    if points[0].features is not None:
        out.features = torch.cat([p.features for p in points], dim=0)
    if points[0].labels is not None:
        out.labels = torch.cat([p.labels for p in points], dim=0)
    out.device = points[0].device

    if update_batch_info:
        out.batch_size = len(points)
        out.batch_npt = torch.Tensor([p.npt for p in points]).long()
        out.batch_id = torch.cat(
            [p.points.new_full((p.npt, 1), i) for i, p in enumerate(points)], dim=0
        ).long()

    return out
