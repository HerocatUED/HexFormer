# utils for data preprocess

import yaml
import torch
import numpy as np
from plyfile import PlyData

from hextree import Points, Hextree


class ReadPly:

    def __init__(
        self, has_normal: bool = True, has_color: bool = False, has_label: bool = False
    ):
        self.has_normal = has_normal
        self.has_color = has_color
        self.has_label = has_label

    def __call__(self, filename: str):
        plydata = PlyData.read(filename)
        vtx = plydata["vertex"]

        output = dict()
        points = np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=1)
        output["points"] = points.astype(np.float32)
        if self.has_normal:
            normal = np.stack([vtx["nx"], vtx["ny"], vtx["nz"]], axis=1)
            output["normals"] = normal.astype(np.float32)
        if self.has_color:
            color = np.stack([vtx["red"], vtx["green"], vtx["blue"]], axis=1)
            output["colors"] = color.astype(np.float32)
        if self.has_label:
            label = vtx["label"]
            output["labels"] = label.astype(np.int32)
        return output


class ReadNpz:

    def __init__(
        self, has_normal: bool = True, has_color: bool = False, has_label: bool = False
    ):
        self.has_normal = has_normal
        self.has_color = has_color
        self.has_label = has_label

    def __call__(self, filename: str):
        raw = np.load(filename)

        output = dict()
        output["points"] = raw["points"].astype(np.float32)
        if self.has_normal:
            output["normals"] = raw["normals"].astype(np.float32)
        if self.has_color:
            output["colors"] = raw["colors"].astype(np.float32)
        if self.has_label:
            output["labels"] = raw["labels"].astype(np.int32)
        return output


class ReadFile:

    def __init__(
        self, has_normal: bool = False, has_color: bool = False, has_label: bool = False
    ):
        self.read_npz = ReadNpz(has_normal, has_color, has_label)
        self.read_ply = ReadPly(has_normal, has_color, has_label)

    def __call__(self, filename: str):
        func = {"npz": self.read_npz, "ply": self.read_ply}
        suffix = filename.split(".")[-1]
        return func[suffix](filename)


class Transform:
    r"""A boilerplate class which transforms an input data.
    The input data is first converted to :class:`Points`, then randomly transformed
    (if enabled), and converted to an :class:`Hextree`.

    Args:
      depth (int): The hextree depth.
      full_depth (int): The hextree layers with a depth small than
          :attr:`full_depth` are forced to be full.
      distort (bool): If true, performs the data augmentation.
      angle (list): A list of 3 float values to generate random rotation angles.
      interval (list): A list of 3 float values to represent the interval of
          rotation angles.
      scale (float): The maximum relative scale factor.
      uniform (bool): If true, performs uniform scaling.
      jittor (float): The maximum jitter values.
      orient_normal (str): Orient point normals along the specified axis, which is
          useful when normals are not oriented.
    """

    # def __init__(self, depth: int, full_depth: int, distort: bool, angle: list,
    #              interval: list, scale: float, uniform: bool, jitter: float,
    #              flip: list, orient_normal: str = '', **kwargs):
    def __init__(
        self,
        depth: int,
        full_depth: int,
        distort: bool,
        angle: list,
        interval: list,
        scale: float,
        flip: list,
        uniform: bool,
        **kwargs
    ):
        super().__init__()

        # for hextree building
        self.depth = depth
        self.full_depth = full_depth

        # for data augmentation
        self.distort = distort
        self.angle = angle
        self.interval = interval
        self.scale = scale
        self.uniform = uniform
        # self.jitter = jitter
        self.flip = flip

        # for other transformations
        # self.orient_normal = orient_normal

    def __call__(self, sample: dict, idx: int):
        """"""

        points = self.preprocess(sample, idx)
        output = self.transform(points, idx)
        output["hextree"] = self.points2hextree(output["points"])
        return output

    def preprocess(self, sample: dict, idx: int):
        r"""
        Transforms :attr:`sample` to :class:`Points` and performs some specific
        transformations, like normalization.
        """

        txyz = torch.from_numpy(sample["points"])
        points = Points(txyz)
        return points

    def transform(self, points: Points, idx: int):
        r"""
        Applies the general transformations.
        """

        # The augmentations including rotation, scaling.
        if self.distort:
            rng_angle, rng_scale, rnd_flip = self.rnd_parameters()

            points.rotate(rng_angle)
            points.scale_xyz(rng_scale)
            points.flip(rnd_flip)

        # if self.orient_normal:
        #     points.orient_normal(self.orient_normal)

        # !!! NOTE: Clip the point cloud to [-1, 1] before building the hextree
        # inbox_mask = points.clip_xyz(bbmin=-1, bbmax=1)
        # return {'points': points, 'inbox_mask': inbox_mask}

        return points

    def points2hextree(self, points: Points):
        r"""
        Converts the input :attr:`points` to an hextree.
        """

        hextree = Hextree(self.depth, self.full_depth)
        hextree.build_hextree(points)
        return hextree

    def rnd_parameters(self):
        r"""
        Generates random parameters for data augmentation.
        """

        rnd_angle = [None] * 3
        for i in range(3):
            rot_num = self.angle[i] // self.interval[i]
            rnd = torch.randint(low=-rot_num, high=rot_num + 1, size=(1,))
            rnd_angle[i] = rnd * self.interval[i] * (torch.pi / 180.0)
        rnd_angle = torch.cat(rnd_angle)

        rnd_scale = torch.rand(3) * (2 * self.scale) - self.scale + 1.0
        if self.uniform:
            rnd_scale[1] = rnd_scale[0]
            rnd_scale[2] = rnd_scale[0]

        rnd_flip = ""
        for i, c in enumerate("xyz"):
            if torch.rand([1]) < self.flip[i]:
                rnd_flip = rnd_flip + c

        return rnd_angle, rnd_scale, rnd_flip
