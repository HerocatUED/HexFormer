import torch
from hextree import Hextree
from .hextree_pad import hextree_pad


def cartesian_to_polar(xyz):
    r"""
    trans cartesian coordinates to polar coordinates
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arctan2(y, x)
    phi = torch.arccos(z / r)
    # theta_deg = np.degrees(theta)
    # phi_deg = np.degrees(phi)
    polar = torch.column_stack((r, theta, phi))
    return polar


class InputFeature(torch.nn.Module):
    r"""Returns the initial input feature stored in hexree.

    Args:
      feature (str): A string used to indicate which features to extract from the
          input hexree.
          If the character :obj:`N` is in :attr:`feature`, the normal signal is extracted (3 channels).
          If :obj:`D` is in :attr:`feature`, the local displacement is extracted (1 channels).
          If :obj:`L` is in :attr:`feature`, the local coordinates of the averaged points in each hexree node is extracted (3 channels).
          If :attr:`G` is in :attr:`feature`, the global coordinates are extracted (4 channels).
          If :attr:`R` is in :attr:`feature`, the relative coordinates are extracted (4 channels).
          If :attr:`P` is in :attr:`feature`, the polar coordinates are extracted (4 channels).
          If :attr:`F` is in :attr:`feature`, other features (like colors) are extracted (k channels).
      nempty (bool): If false, gets the features of all hexree nodes.
    """

    def __init__(self, feature: str = "NDF", nempty: bool = False):
        super().__init__()
        self.nempty = nempty
        self.feature = feature.upper()

    def forward(self, hexree: Hextree):
        r""""""

        features = list()
        depth = hexree.depth
        if "N" in self.feature:
            features.append(hexree.normals[depth])

        if "L" in self.feature or "D" in self.feature:
            local_points = hexree.points[depth].frac() - 0.5

        if "D" in self.feature:
            dis = torch.sum(local_points * hexree.normals[depth], dim=1, keepdim=True)
            features.append(dis)

        if "L" in self.feature:
            features.append(local_points)

        if "G" in self.feature:
            scale = 2 ** (1 - depth)  # normalize xyz [0, 2^depth] -> [-1, 1]
            global_points = hexree.points[depth] * scale - 1.0
            # normalize t -> 0, ... not [-1, 1]
            global_points[:, 0] = hexree.points[depth][:, 0] - torch.min(
                hexree.points[depth][:, 0]
            )
            features.append(global_points)

        if "P" in self.feature:
            polar_points = cartesian_to_polar(global_points[:, 1:])
            features.append(polar_points)

        if "F" in self.feature:
            features.append(hexree.features[depth])

        out = torch.cat(features, dim=1)
        if not self.nempty:
            out = hextree_pad(out, hexree, depth)
        return out

    # def extra_repr(self) -> str:
    #     r""""""
    #     return "feature={}, nempty={}".format(self.feature, self.nempty)
