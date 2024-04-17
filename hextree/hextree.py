import torch
from typing import Union, List

import ocnn
from .utils import cumsum
from .points import Points
from .shuffled_key import txyz2key, key2txyz


class Hextree:
    r"""Builds an hextree from an input pointcloud.

    Args:
        depth (int): The hextree depth.
        full_depth (int): The hextree layers with a depth small than
            :attr:`full_depth` are forced to be full.
        batch_size (int): The hextree batch size. If you build the tree from Points,
            please set batch_size to 1. If you merge multiple trees by calling
            :func:`merge_hextrees`, batch_size will be automatically set.
        device (torch.device or str): Choose from :obj:`cpu` and :obj:`gpu`.
            (default: :obj:`cpu`)

    .. note::
        The hextree data structure requires that if an hextree node has children nodes,
        the number of children nodes is exactly 16, in which some of the nodes are
        empty and some nodes are non-empty. The properties of an hextree, including
        :obj:`keys`, :obj:`children` and :obj:`neighs`, contain both non-empty and
        empty nodes, and other properties, including :obj:`features`, :obj:`normals`
        and :obj:`points`, contain only non-empty nodes.

    .. note::
        The point cloud must be in range :obj:`[-1, 1]`.
        
    .. note::
        t is dense. e.g. if point_cloud has unique_t = [0, 2, 90], it will be taken as [0, 1, 2]
    """

    def __init__(
        self,
        depth: int,
        full_depth: int = 2,
        batch_size: int = 1,
        device: Union[torch.device, str] = "cpu",
        **kwargs
    ):
        super().__init__()
        self.depth = depth
        self.full_depth = full_depth
        self.batch_size = batch_size
        self.device = device

        self.reset()

    def nempty_mask(self, depth: int):
        r"""Returns a binary mask (Tensor[bool]) which indicates whether the cooreponding
        hextree node is empty or not.

        Args:
            depth (int): The depth of the hextree.
        """

        return self.children[depth] >= 0

    # properties
    def key(self, depth: int, nempty: bool = False):
        r"""Returns the shuffled key of each hextree node.

        Args:
            depth (int): The depth of the hextree.
            nempty (bool): If True, returns the results of non-empty hextree nodes.
        """

        key = self.keys[depth]
        if nempty:
            mask = self.nempty_mask(depth)
            key = key[mask]
        return key

    def txyzb(self, depth: int, nempty: bool = False):
        r"""Returns the txyz coordinates and the batch indices of each hextree node.

        Args:
            depth (int): The depth of the hextree.
            nempty (bool): If True, returns the results of non-empty hextree nodes.
        """

        key = self.key(depth, nempty)
        return key2txyz(key, depth)

    def batch_id(self, depth: int, nempty: bool = False):
        r"""Returns the batch indices of each hextree node.

        Args:
            depth (int): The depth of the hextree.
            nempty (bool): If True, returns the results of non-empty hextree nodes.
        """

        batch_id = self.key(depth, nempty) >> 56
        return batch_id

    def search_txyzb(self, query: torch.Tensor, depth: int, nempty: bool = False):
        r"""Searches the hextree nodes given the query points. Corresponding element
        is -1 if a point cannot be found

        Args:
          query (torch.Tensor): The coordinates of query points with shape
              :obj:`(N, 5)`. The first 4 channels of the coordinates are :obj:`t`,
              :obj:`x`, :obj:`y`, and :obj:`z`, and the last channel is the batch
              index. Note that the coordinates must be in range :obj:`[0, 2^depth)`.
          depth (int): The depth of the hextree layer.
          nemtpy (bool): If true, only searches the non-empty hextree nodes.
        """

        key = txyz2key(
            query[:, 0], query[:, 1], query[:, 2], query[:, 3], query[:, 4], depth
        )
        idx = self.search_key(key, depth, nempty)
        return idx

    def search_key(self, query: torch.Tensor, depth: int, nempty: bool = False):
        r"""Searches the hextree nodes given the query points. Corresponding element
        is -1 if a point cannot be found

        Args:
        query (torch.Tensor): The keys of query points with shape :obj:`(N,)`,
            which are computed from the coordinates of query points.
        depth (int): The depth of the hextree layer. nemtpy (bool): If true, only
            searches the non-empty hextree nodes.
        """

        key = self.key(depth, nempty)
        # `torch.bucketize` is similar to `torch.searchsorted`.
        # I choose `torch.bucketize` here because it has fewer dimension checks,
        # resulting in slightly better performance according to the docs of
        # pytorch-1.9.1, since `key` is always 1-D sorted sequence.

        idx = torch.bucketize(query, key)

        valid = idx < key.shape[0]  # invalid if out of bound
        found = key[idx[valid]] == query[valid]
        valid[valid.clone()] = found
        idx[valid.logical_not()] = -1
        return idx

    # builing the tree
    def reset(self):
        r"""Resets the Hextree status and constructs several lookup tables."""

        # hextree features in each hextree layers
        num = self.depth + 1
        self.keys = [None] * num
        self.children = [None] * num
        self.features = [None] * num
        self.normals = [None] * num
        self.points = [None] * num
        
        self.octrees = None
        self.octree_list = []
        # mapping index between hextree and octree
        self.hex2oct = [None] * num
        self.oct2hex = [None] * num
        self.hex2oct_nempty = [None] * num
        self.oct2hex_nempty = [None] * num

        # hextree node numbers in each hextree layers
        # TODO: decide whether to settle them to 'gpu' or not
        self.nnum = torch.zeros(num, dtype=torch.int64)
        self.nnum_nempty = torch.zeros(num, dtype=torch.int64)

        # the following properties are valid after `merge_hextrees`
        # TODO: make them valid after `hextree_grow`, `hextree_split` and `build_hextree`
        batch_size = self.batch_size
        self.batch_nnum = torch.zeros(num, batch_size, dtype=torch.int64)
        self.batch_nnum_nempty = torch.zeros(num, batch_size, dtype=torch.int64)

    def build_hextree(self, point_cloud: Points):
        r"""Builds a hextree from a point cloud.

        Args:
            point_cloud (Points): The input point cloud, xyz in [-1, 1], t is int.

        .. note::
            Currently, the batch size of the point cloud must be 1.
        """

        self.device = point_cloud.device
        assert point_cloud.batch_size == self.batch_size, \
            "Inconsistent batch_size, only supported when building hextree!"

        # build octree frame by frame
        points, normals, features = point_cloud.points, point_cloud.normals, point_cloud.features
        t, x, y, z = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        assert torch.max(t) < 2 ** 8 and torch.min(t) >= 0, \
            "t should be in range [0, 255]"
        assert torch.max(x) <= 1 and torch.min(x) <= 1 and \
                torch.max(x) <= 1 and torch.min(x) <= 1 and \
                torch.max(x) <= 1 and torch.min(x) <= 1 ,\
            "You should normalize xyz to [-1, 1] before build tree."
        t = t.long()
        for i in torch.unique(t): 
            normal = normals[mask] if normals is not None else None
            feature = features[mask] if features is not None else None
            mask = t == i
            pts = torch.concatenate([x[mask].unsqueeze(1), y[mask].unsqueeze(1), z[mask].unsqueeze(1)], dim=1)
            pts = ocnn.octree.Points(pts, normal, feature)
            otree = ocnn.octree.Octree(self.depth, self.full_depth)
            otree.build_octree(pts)
            self.octree_list.append(otree)
        self.octrees = ocnn.octree.merge_octrees(self.octree_list)
        self.octrees.construct_all_neigh()
        
        # build hextree:
        for d in range(self.depth, -1, -1):
            # key
            okey = self.octrees.keys[d]
            hkey = ((okey << 16) >> 8) | ((okey << 8) >> 56)
            self.keys[d], idx = torch.sort(hkey)
            self.oct2hex[d] = idx
            _, self.hex2oct[d] = torch.sort(idx)
            self.nnum[d] = okey.numel()
            # key_nempty
            okey_nempty = self.octrees.key(d, True)
            hkey_nempty = ((okey_nempty << 16) >> 8) | ((okey_nempty << 8) >> 56)
            _, idx_nempty = torch.sort(hkey_nempty)
            self.oct2hex_nempty[d] = idx_nempty
            _, self.hex2oct_nempty[d] = torch.sort(idx_nempty)
            self.nnum_nempty[d] = okey_nempty.numel()
            # children
            self.children[d] = self.octrees.children[d][self.oct2hex[d]]

        # average the signal for the last hextree layer
        d = self.depth
        self.points[d] = self.octrees.points[d][self.oct2hex_nempty[d]]
        if self.octrees.normals[d] is not None:
            self.normals[d] = self.octrees.normals[d][self.oct2hex_nempty[d]]
        if self.octrees.features[d] is not None:
            self.features[d] = self.octrees.features[d][self.oct2hex_nempty[d]]

    # tree to points transformation
    def get_input_feature(self):
        r"""Gets the initial input features.

        .. note::
            Correctness not guaranteed since it is not used in our model
        """

        # normals
        features = list()
        depth = self.depth
        has_normal = self.normals[depth] is not None
        if has_normal:
            features.append(self.normals[depth])

        # local points
        points = self.points[depth][:, 1:].frac() - 0.5
        if has_normal:
            dis = torch.sum(points * self.normals[depth], dim=1, keepdim=True)
            features.append(dis)
        else:
            features.append(points)

        # features
        if self.features[depth] is not None:
            features.append(self.features[depth])

        return torch.cat(features, dim=1)

    def to_points(self, rescale: bool = True):
        r"""Converts averaged points in the hextree to a pointcloud.

        Args:
            rescale (bool): rescale the xyz coordinates to [-1, 1] if True
        """

        depth = self.depth
        batch_size = self.batch_size

        # by default, use the average points generated when building the hextree
        # from the input point cloud
        txyz = self.points[depth]
        batch_id = self.batch_id(depth, nempty=True)

        # txyz is None when the hextree is predicted by a neural network
        if txyz is None:
            t, x, y, z, batch_id = self.txyzb(depth, nempty=True)
            txyz = torch.stack([t, x, y, z], dim=1)
            txyz[:, 1:] += 0.5

        # normalize xyz to [-1, 1] since the average points are in range [0, 2 ^ d]
        if rescale:
            scale = 2 ** (1 - depth)
            txyz = self.points[depth].copy()
            txyz[:, 1:] = txyz[:, 1:] * scale - 1.0

        # construct Points
        out = Points(
            txyz,
            self.normals[depth],
            self.features[depth],
            batch_id=batch_id,
            batch_size=batch_size,
        )
        return out

    # torch device operation
    def to(self, device: Union[torch.device, str], non_blocking: bool = False):
        r"""Moves the hextree to a specified device.

        Args:
          device (torch.device or str): The destination device.
          non_blocking (bool): If True and the source is in pinned memory, the copy
              will be asynchronous with respect to the host. Otherwise, the argument
              has no effect. Default: False.
        """

        if isinstance(device, str):
            device = torch.device(device)

        #  If on the save device, directly retrun self
        if self.device == device:
            return self

        def list_to_device(prop):
            return [
                (
                    p.to(device, non_blocking=non_blocking)
                    if isinstance(p, torch.Tensor)
                    else None
                )
                for p in prop
            ]

        # Construct a new hextree on the specified device
        hextree = Hextree(self.depth, self.full_depth, self.batch_size, device)
        hextree.keys = list_to_device(self.keys)
        hextree.children = list_to_device(self.children)
        hextree.features = list_to_device(self.features)
        hextree.normals = list_to_device(self.normals)
        hextree.points = list_to_device(self.points)
        hextree.oct2hex = list_to_device(self.oct2hex)
        hextree.hex2oct = list_to_device(self.hex2oct)
        hextree.oct2hex_nempty = list_to_device(self.oct2hex_nempty)
        hextree.hex2oct_nempty = list_to_device(self.hex2oct_nempty)
        hextree.nnum = (self.nnum.clone())  # TODO: whether to move nnum to the self.device?
        hextree.nnum_nempty = self.nnum_nempty.clone()
        hextree.batch_nnum = self.batch_nnum.clone()
        hextree.batch_nnum_nempty = self.batch_nnum_nempty.clone()
        hextree.octrees = self.octrees.to(device)
        hextree.octree_list = list_to_device(self.octree_list)
        return hextree

    def cuda(self, non_blocking: bool = False):
        r"""Moves the hextree to the GPU."""

        return self.to("cuda", non_blocking)

    def cpu(self):
        r"""Moves the hextree to the CPU."""

        return self.to("cpu")


def merge_hextrees(hextrees: List["Hextree"]):
    r"""Merges a list of hextrees (batch_size = 1) into one batch.

    Args:
        hextrees (List[Hextree]): A list of hextrees to merge.
    """

    # init and check
    hextree = Hextree(
        depth=hextrees[0].depth,
        full_depth=hextrees[0].full_depth,
        batch_size=len(hextrees),
        device=hextrees[0].device,
    )
    for i in range(1, hextree.batch_size):
        condition = (
            hextrees[i].depth == hextree.depth
            and hextrees[i].full_depth == hextree.full_depth
            and hextrees[i].device == hextree.device
        )
        assert condition, "The check of merge_hextrees failed"

    # node num
    batch_nnum = torch.stack(
        [hextrees[i].nnum for i in range(hextree.batch_size)], dim=1
    )
    batch_nnum_nempty = torch.stack(
        [hextrees[i].nnum_nempty for i in range(hextree.batch_size)], dim=1
    )
    hextree.nnum = torch.sum(batch_nnum, dim=1)
    hextree.nnum_nempty = torch.sum(batch_nnum_nempty, dim=1)
    hextree.batch_nnum = batch_nnum
    hextree.batch_nnum_nempty = batch_nnum_nempty
    nnum_cum = cumsum(batch_nnum, dim=1, exclusive=True)
    nnum_cum_nempty = cumsum(batch_nnum_nempty, dim=1, exclusive=True)

    # merge hextree properties
    for d in range(hextree.depth + 1):
        # key
        keys = [None] * hextree.batch_size
        for i in range(hextree.batch_size):
            key = hextrees[i].keys[d] & ((1 << 56) - 1)  # clear the highest bits
            keys[i] = key | (i << 56)
        hextree.keys[d] = torch.cat(keys, dim=0)

        # children and mapping index
        children = [None] * hextree.batch_size
        hex2oct = [None] * hextree.batch_size
        oct2hex = [None] * hextree.batch_size
        hex2oct_nempty = [None] * hextree.batch_size
        oct2hex_nempty = [None] * hextree.batch_size
        for i in range(hextree.batch_size):
            # !! `clone` is used here to avoid
            child = hextrees[i].children[d].clone()
            mask = child >= 0  # !! modifying the original hextrees
            child[mask] = child[mask] + nnum_cum_nempty[d, i]
            children[i] = child
            # mapping index nempty
            idx_ho = hextrees[i].hex2oct_nempty[d].clone()
            idx_ho += nnum_cum_nempty[d, i]
            hex2oct_nempty[i] = idx_ho
            idx_oh = hextrees[i].oct2hex_nempty[d].clone()
            idx_oh += nnum_cum_nempty[d, i]
            oct2hex_nempty[i] = idx_oh
            # mapping index nempty
            idx_ho = hextrees[i].hex2oct[d].clone()
            idx_ho += nnum_cum[d, i]
            hex2oct[i] = idx_ho
            idx_oh = hextrees[i].oct2hex[d].clone()
            idx_oh += nnum_cum[d, i]
            oct2hex[i] = idx_oh
            
        hextree.children[d] = torch.cat(children, dim=0)
        hextree.hex2oct[d] = torch.cat(hex2oct, dim=0)
        hextree.oct2hex[d] = torch.cat(oct2hex, dim=0)
        hextree.hex2oct_nempty[d] = torch.cat(hex2oct_nempty, dim=0)
        hextree.oct2hex_nempty[d] = torch.cat(oct2hex_nempty, dim=0)

    d = hextree.depth
    # features
    if hextrees[0].features[d] is not None:
        features = [hextrees[i].features[d] for i in range(hextree.batch_size)]
        hextree.features[d] = torch.cat(features, dim=0)

    # normals
    if hextrees[0].normals[d] is not None:
        normals = [hextrees[i].normals[d] for i in range(hextree.batch_size)]
        hextree.normals[d] = torch.cat(normals, dim=0)

    # points
    if hextrees[0].points[d] is not None:
        points = [hextrees[i].points[d] for i in range(hextree.batch_size)]
        hextree.points[d] = torch.cat(points, dim=0)
    
    # octrees
    for i in range(hextree.batch_size):
        hextree.octree_list += hextrees[i].octree_list
    hextree.octrees = ocnn.octree.merge_octrees(hextree.octree_list)
    hextree.octrees.construct_all_neigh()
    
    return hextree
