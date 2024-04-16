import torch
import torch.nn.functional as F
from typing import Union, List

import ocnn
from .utils import meshgrid, scatter_add, cumsum
from .points import Points
from .shuffled_key import txyz2key, key2txyz
from ocnn.octree.shuffled_key import xyz2key


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

    # utils
    def rng_grid(self, min_, max_):
        r"""Builds a 4D grid in :obj:`[min, max]` (:attr:`max` included)."""

        rng = torch.arange(min_, max_ + 1, dtype=torch.long, device=self.device)
        grid = meshgrid(rng, rng, rng, rng, indexing="ij")
        # ((max_ - min_) ** 4, 4)
        grid = torch.stack(grid, dim=-1).view(-1, 4)
        return grid

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

        # mapping index between hextree and octree
        # valid only after merge hextree
        self.hextree2octree = [None] * num
        self.octree2hextree = [None] * num
        self.octrees = None

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
            point_cloud (Points): The input point cloud.

        .. note::
            Currently, the batch size of the point cloud must be 1.
        """

        self.device = point_cloud.device
        assert point_cloud.batch_size == self.batch_size, "Inconsistent batch_size"

        # normalize points from [-1, 1] to [0, 2 ^ depth)
        scale = 1 << (self.depth - 1)
        ps = point_cloud.points
        points = torch.cat(
            [ps[:, [0]].long(), ((ps[:, 1:] + 1.0) * scale).long()], dim=1
        )
        points[points == 2 * scale] = 2 * scale - 1  # 2 ^ depth -> 2 ^ depth - 1

        # check t
        tmax = torch.max(points[:, 0])
        assert tmax < (2 << 8 - 1)

        # get the shuffled key and sort
        t, x, y, z = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        # TODO: Allow multiple batches
        b = None if self.batch_size == 1 else point_cloud.batch_id.view(-1)
        key = txyz2key(t, x, y, z, b, self.depth)
        node_key, idx, counts = torch.unique(
            key, sorted=True, return_inverse=True, return_counts=True, dim=0
        )

        # layer 0 to full_layer: the hextree is full in these layers
        for d in range(self.full_depth + 1):
            self.hextree_grow_full(d, update_neigh=False)

        # layer depth to full_layer
        t = node_key & (2**8 - 1)
        node_key = t << 56 | node_key >> 8  # tb(xyz)
        for d in range(self.depth, self.full_depth, -1):
            # compute parent key, i.e. keys of layer (d-1)
            pkey = node_key >> 3
            pkey, pidx = torch.unique_consecutive(pkey, return_inverse=True, dim=0)

            # augmented key
            key = (
                (pkey.unsqueeze(-1) << 3) + torch.arange(9, device=self.device)
            ).view(-1)
            self.keys[d] = key >> 56 | key << 8
            self.nnum[d] = key.numel()
            self.nnum_nempty[d] = node_key.numel()

            # children
            addr = (pidx << 3) | (node_key % 8)
            children = -torch.ones(self.nnum[d], dtype=torch.int64, device=self.device)
            children[addr] = torch.arange(
                self.nnum_nempty[d], dtype=torch.int64, device=self.device
            )
            self.children[d] = children

            # cache pkey for the next iteration
            node_key = pkey

        # build mapping index
        self.build_mapping_idx()
        self.build_octrees()

        # set the children for the layer full_layer,
        # now the node_keys are the key for full_layer
        d = self.full_depth
        children = -torch.ones_like(self.children[d], dtype=torch.int64)
        nempty_idx = (
            node_key
            if self.batch_size == 1
            else ((node_key >> 48) << (3 * d)) | (node_key * ((1 << 48) - 1))
        )
        children[nempty_idx] = torch.arange(
            node_key.numel(), dtype=torch.int64, device=self.device
        )
        self.children[d] = children
        self.nnum_nempty[d] = node_key.numel()

        # average the signal for the last hextree layer
        d = self.depth
        # points is rescaled in [L:Scale]
        points = scatter_add(points, idx, dim=0)
        self.points[d] = points / counts.unsqueeze(1)
        if point_cloud.normals is not None:
            normals = scatter_add(point_cloud.normals, idx, dim=0)
            self.normals[d] = F.normalize(normals)
        if point_cloud.features is not None:
            features = scatter_add(point_cloud.features, idx, dim=0)
            self.features[d] = features / counts.unsqueeze(1)

        return idx

    def hextree_grow_full(self, depth: int):
        r"""Builds the full hextree, which is essentially a dense volumetric grid.

        Args:
            depth (int): The depth of the hextree.
        """

        # check
        assert depth <= self.full_depth, "error"

        # node number
        num = 1 << (4 * depth)
        self.nnum[depth] = num * self.batch_size
        self.nnum_nempty[depth] = num * self.batch_size

        # update key
        key = torch.arange(num, dtype=torch.long, device=self.device)
        bs = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        key = key.unsqueeze(0) | (bs.unsqueeze(1) << 56)
        self.keys[depth] = key.view(-1)

        # update children
        self.children[depth] = torch.arange(
            num * self.batch_size, dtype=torch.int64, device=self.device
        )

    def build_mapping_idx(self):
        r"""Sets attributes `hextree2octree` and `octree2hextree`"""

        for d in range(self.depth, -1, -1):
            hextree_key = self.key(d, nempty=True)
            t, x, y, z, b = key2txyz(hextree_key, d)
            bt = t.clone()
            cnt = 0
            for i in torch.unique(b, sorted=True):
                mask = b == i
                bt[mask] += cnt
                cnt += t[mask].max() + 1
            octree_keys = xyz2key(x, y, z, bt, d)
            _, idx = torch.sort(octree_keys)
            self.octree2hextree[d] = idx  # hextree key idx
            _, self.hextree2octree[d] = torch.sort(idx)  # octree key idx

    def build_octrees(self):
        otrees = []
        hextree_key = self.key(self.depth, nempty=True)
        t, x, y, z, b = key2txyz(hextree_key, self.depth)
        for i in torch.unique(b, sorted=True):
            mask = b == i
            for j in torch.unique(t[mask], sorted=True):
                mask1 = (t == j) & mask
                ox = x[mask1].unsqueeze(1)
                oy = y[mask1].unsqueeze(1)
                oz = z[mask1].unsqueeze(1)
                pts = torch.concatenate([ox, oy, oz], dim=1)
                # normalize points from  [0, 2^depth] to [-1, 1]
                scale = 2 ** (self.depth - 1)
                pts = pts / scale - 1
                points = ocnn.octree.Points(pts)
                otree = ocnn.octree.Octree(self.depth, self.full_depth)
                otree.build_octree(points)
                otrees.append(otree)
        self.octrees = ocnn.octree.merge_octrees(otrees)
        self.octrees.construct_all_neigh()

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
        hextree.octree2hextree = list_to_device(self.octree2hextree)
        hextree.hextree2octree = list_to_device(self.hextree2octree)
        hextree.nnum = (
            self.nnum.clone()
        )  # TODO: whether to move nnum to the self.device?
        hextree.nnum_nempty = self.nnum_nempty.clone()
        hextree.batch_nnum = self.batch_nnum.clone()
        hextree.batch_nnum_nempty = self.batch_nnum_nempty.clone()
        hextree.octrees = self.octrees.to(device)
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
    nnum_cum = cumsum(batch_nnum_nempty, dim=1, exclusive=True)

    # merge hextree properties
    for d in range(hextree.depth + 1):
        # key
        keys = [None] * hextree.batch_size
        for i in range(hextree.batch_size):
            key = hextrees[i].keys[d] & ((1 << 56) - 1)  # clear the highest bits
            keys[i] = key | (i << 56)
        hextree.keys[d] = torch.cat(keys, dim=0)

        # children
        children = [None] * hextree.batch_size
        for i in range(hextree.batch_size):
            # !! `clone` is used here to avoid
            child = hextrees[i].children[d].clone()
            mask = child >= 0  # !! modifying the original hextrees
            child[mask] = child[mask] + nnum_cum[d, i]
            children[i] = child
        hextree.children[d] = torch.cat(children, dim=0)

        # features
        if hextrees[0].features[d] is not None and d == hextree.depth:
            features = [hextrees[i].features[d] for i in range(hextree.batch_size)]
            hextree.features[d] = torch.cat(features, dim=0)

        # normals
        if hextrees[0].normals[d] is not None and d == hextree.depth:
            normals = [hextrees[i].normals[d] for i in range(hextree.batch_size)]
            hextree.normals[d] = torch.cat(normals, dim=0)

        # points
        if hextrees[0].points[d] is not None and d == hextree.depth:
            points = [hextrees[i].points[d] for i in range(hextree.batch_size)]
            hextree.points[d] = torch.cat(points, dim=0)

    # mapping index between hextree and octree
    num = hextree.depth + 1
    hextree.hextree2octree = [None] * num
    hextree.octree2hextree = [None] * num
    hextree.build_mapping_idx()
    hextree.build_octrees()
    return hextree
