# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# Hextree modified by Ruihuan Wang
# --------------------------------------------------------

import torch 
import torch.nn.functional as F
from typing import Union, List

import sys 
sys.path.append('..')
from .utils import meshgrid, scatter_add, cumsum, trunc_div
from .points import Points
from .shuffled_key import txyz2key, key2txyz


class Hextree:
    r''' Builds an hextree from an input pointcloud.

    Args:
        depth (int): The hextree depth.
        full_depth (int): The hextree layers with a depth small than
            :attr:`full_depth` are forced to be full.
        batch_size (int): The hextree batch size.
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
    '''

    def __init__(self, depth: int, full_depth: int = 2, batch_size: int = 1,
                 device: Union[torch.device, str] = 'cpu', **kwargs):
        super().__init__()
        self.depth = depth
        self.full_depth = full_depth 
        self.batch_size = batch_size
        self.device = device

        self.reset()
    
    def reset(self):
        r''' Resets the Hextree status and constructs several lookup tables.
        '''

        # hextree features in each hextree layers
        num = self.depth + 1
        self.keys = [None] * num 
        self.children = [None] * num 
        self.neighs = [None] * num 
        self.features = [None] * num 
        self.normals = [None] * num 
        self.points = [None] * num 

        # hextree node numbers in each hextree layers
        # TODO: decide whether to settle them to 'gpu' or not
        self.nnum = torch.zeros(num, dtype=torch.int64)
        self.nnum_nempty = torch.zeros(num, dtype=torch.int64)

        # the following properties are valid after `merge_hextrees`
        # TODO: make them valid after `hextree_grow`, `hextree_split` and `build_hextree`
        batch_size = self.batch_size
        self.batch_nnum = torch.zeros(num, batch_size, dtype=torch.int64)
        self.batch_nnum_nempty = torch.zeros(num, batch_size, dtype=torch.int64)

        # construct the look up tables for neighbourhood searching
        device = self.device
        center_grid = self.rng_grid(2, 3)                       # (16, 4)
        displacement = self.rng_grid(-1, 1)                     # (81, 4)
        neigh_grid = center_grid.unsqueeze(1) + displacement    # (16, 81, 4)
        parent_grid = trunc_div(neigh_grid, 2)                  # (16, 81, 4)
        child_grid = neigh_grid % 2                             # (16, 81, 4)                   
        self.lut_parent = torch.sum(
            parent_grid * torch.tensor([27, 9, 3, 1], device=device), dim=2)    # (16, 81)
        # For a certain node N, 
        # N.neigh[lut_parent[i, j]].index = N.child[i].neigh[j].parent.index
        self.lut_child = torch.sum(
            child_grid * torch.tensor([8, 4, 2, 1], device=device), dim=2)     # (16, 81)
        # For a certain node N,
        # N.child[i].neigh[j].index = N.child[i].neigh[j].parent.index * 16 + lut_child[i, j]

        # lookup tables for different kernel sizes
        self.lut_kernel = {}
    
    def key(self, depth: int, nempty: bool = False):
        r''' Returns the shuffled key of each hextree node.

        Args:
            depth (int): The depth of the hextree.
            nempty (bool): If True, returns the results of non-empty hextree nodes.
        '''

        key = self.keys[depth]
        if nempty:
            mask = self.nempty_mask(depth)
            key = key[mask]
        return key
    
    def txyzb(self, depth: int, nempty: bool = False):
        r''' Returns the xyz coordinates and the batch indices of each hextree node.

        Args:
            depth (int): The depth of the hextree.
            nempty (bool): If True, returns the results of non-empty hextree nodes.
        '''

        key = self.key(depth, nempty)
        return key2txyz(key, depth)
    
    def batch_id(self, depth: int, nempty: bool = False):
        r''' Returns the batch indices of each hextree node.

        Args:
            depth (int): The depth of the hextree.
            nempty (bool): If True, returns the results of non-empty hextree nodes.
        '''

        batch_id = self.keys[depth][:, 0]
        if nempty:
            mask = self.nempty_mask(depth)
            batch_id = batch_id[mask]
        return batch_id

    def nempty_mask(self, depth: int):
        r''' Returns a binary mask which indicates whether the cooreponding hextree
        node is empty or not.

        Args:
            depth (int): The depth of the hextree.
        '''

        return self.children[depth] >= 0

    def build_hextree(self, point_cloud: Points):
        r''' Builds a hextree from a point cloud.

        Args:
            point_cloud (Points): The input point cloud.

        .. note::
            Currently, the batch size of the point cloud must be 1.
        '''

        self.device = point_cloud.device
        assert point_cloud.batch_size == self.batch_size, 'Inconsistent batch_size'

        # normalize points from [-1, 1] to [0, 2 ^ depth]
        scale = 2 ** (self.depth - 1)
        ps = point_cloud.points
        points = torch.cat([ps[:, [0]].long(), ((ps[:, 1:] + 1.0) * scale).long()], dim=1)
        
        # Scaling of t
        tmax = torch.max(points[:, 0])
        # if (scale >> 1) < tmax:
        #     points[:, 0] = points[:, 0] / tmax * scale
        assert tmax <= (scale << 1)

        # get the shuffled key and sort
        t, x, y, z = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        b = None if self.batch_size == 1 else point_cloud.batch_id.view(-1)
        key = txyz2key(t, x, y, z, b, self.depth)
        node_key, idx, counts = torch.unique(
            key, sorted=True, return_inverse=True, return_counts=True, dim=0)
        
        # layer 0 to full_layer: the hextree is full in these layers
        for d in range(self.full_depth + 1):
            self.hextree_grow_full(d, update_neigh=False)
        
        # layer depth to full_layer
        for d in range(self.depth, self.full_depth, -1):
            # compute parent key, i.e. keys of layer (d-1)
            pkey = torch.stack((node_key[...,0], node_key[...,1] >> 4), axis=-1)
            pkey, pidx, pcounts = torch.unique_consecutive(
                pkey, return_inverse=True, return_counts=True, dim=0)

            # augmented key
            key_txyz = (pkey[...,1].unsqueeze(-1) << 4) + torch.arange(16, device=self.device)
            key = torch.stack((pkey[...,0].unsqueeze(-1) + torch.zeros(16, dtype=torch.long), key_txyz), axis=-1)
            self.keys[d] = key.view(-1, 2)
            self.nnum[d] = key[..., 1].numel()
            self.nnum_nempty[d] = node_key[..., 1].numel()

            # children
            addr = (pidx << 4) | (node_key[..., 1] % 16)
            children = -torch.ones(
                self.nnum[d].item(), dtype=torch.int64, device=self.device)
            children[addr] = torch.arange(
                self.nnum_nempty[d], dtype=torch.int64, device=self.device)
            self.children[d] = children

            # cache pkey for the next iteration
            node_key = pkey

        # set the children for the layer full_layer,
        # now the node_keys are the key for full_layer
        d = self.full_depth
        children = -torch.ones_like(self.children[d], dtype=torch.int64)
        # ??? original code << (3 * d) here, dont know why
        nempty_idx = node_key[..., 1]
        children[nempty_idx] = torch.arange(
            node_key[..., 1].numel(), dtype=torch.int64, device=self.device)
        self.children[d] = children
        self.nnum_nempty[d] = node_key[..., 1].numel()

        # average the signal for the last hextree layer
        d = self.depth
        points = scatter_add(points, idx, dim=0)  # points is rescaled in [L:Scale]
        self.points[d] = points / counts.unsqueeze(1)
        if point_cloud.normals is not None:
            normals = scatter_add(point_cloud.normals, idx, dim=0)
            self.normals[d] = F.normalize(normals)
        if point_cloud.features is not None:
            features = scatter_add(point_cloud.features, idx, dim=0)
            self.features[d] = features / counts.unsqueeze(1)

        return idx
    
    def hextree_grow_full(self, depth: int, update_neigh: bool = True):
        r''' Builds the full hextree, which is essentially a dense volumetric grid.

        Args:
            depth (int): The depth of the hextree.
            update_neigh (bool): If True, construct the neighborhood indices.
        '''

        # check
        assert depth <= self.full_depth, 'error'

        # node number
        num = 1 << (4 * depth)
        self.nnum[depth] = num * self.batch_size
        self.nnum_nempty[depth] = num * self.batch_size

        # update key
        key = torch.arange(num, dtype=torch.long, device=self.device)
        bs = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        key = torch.cartesian_prod(bs, key).long()
        self.keys[depth] = key.view(-1, 2)

        # update children
        self.children[depth] = torch.arange(
            num * self.batch_size, dtype=torch.int64, device=self.device)

        # update neigh if needed
        if update_neigh:
            self.construct_neigh(depth)

    def hextree_split(self, split: torch.Tensor, depth: int):
        r''' Sets whether the hextree nodes in :attr:`depth` are splitted or not.

        Args:
            split (torch.Tensor): The input tensor with its element indicating status
                of each hextree node: 0 - empty, 1 - non-empty or splitted.
            depth (int): The depth of current hextree.
        '''

        # split -> children
        empty = split == 0
        sum_ = cumsum(split, dim=0, exclusive=True)
        children, nnum_nempty = torch.split(sum_, [split.shape[0], 1])
        children[empty] = -1

        # boundary case, make sure that at least one hextree node is splitted
        if nnum_nempty == 0:
            nnum_nempty = 1
            children[0] = 0

        # update hextree
        self.children[depth] = children
        self.nnum_nempty[depth] = nnum_nempty

    def hextree_grow(self, depth: int, update_neigh: bool = True):
        r''' Grows the hextree and updates the relevant properties. And in most
        cases, call :func:`Hextree.hextree_split` to update the splitting status of
        the hextree before this function.

        Args:
          depth (int): The depth of the hextree.
          update_neigh (bool): If True, construct the neighborhood indices.
        '''

        # node number
        nnum = self.nnum_nempty[depth-1] * 8
        self.nnum[depth] = nnum
        self.nnum_nempty[depth] = nnum

        # update keys
        key = self.key(depth-1, nempty=True)
        batch_id = key[..., 0]
        key = key[..., -1] << 4
        key = key.unsqueeze(1) + torch.arange(16, device=key.device)
        self.keys[depth] = torch.cat((batch_id, key), axis=-1).view(-1, 2)

        # update children
        self.children[depth] = torch.arange(
            nnum, dtype=torch.int64, device=self.device)

        # update neighs
        if update_neigh:
            self.construct_neigh(depth)

    def construct_neigh(self, depth: int):
        r''' Constructs the :obj:`3x3x3x3` neighbors for each hextree node.

        Args:
            depth (int): The hextree depth with a value larger than 0 (:obj:`>0`).
        '''

        if depth <= self.full_depth:
            nnum = 1 << (4 * depth)
            key = torch.arange(nnum, dtype=torch.long, device=self.device)
            b = torch.zeros_like(key)
            key = torch.stack([b, key], dim=-1)
            t, x, y, z, _ = key2txyz(key, depth)
            txyz = torch.stack([t, x, y, z], dim=-1)  # (N, 4)
            grid = self.rng_grid(min_=-1, max_=1)   # (81, 4)
            txyz = txyz.unsqueeze(1) + grid         # (N, 81, 4)
            txyz = txyz.view(-1, 4)                 # (N * 81, 4)
            neigh = txyz2key(txyz[:, 0], txyz[:, 1], txyz[:, 2], txyz[:, 3], depth=depth)   # (N * 81, 2)

            bs = torch.arange(self.batch_size, dtype=torch.int64, device=self.device)
            neigh = neigh.unsqueeze(0) + bs.unsqueeze(1).unsqueeze(1) * nnum  # (1, N * 81, 2) + (B, 1) -> (B, N * 81, 2)

            bound = 1 << depth
            invalid = torch.logical_or((txyz < 0).any(1), (txyz >= bound).any(1))
            neigh[:, invalid, 1] = -1
            self.neighs[depth] = neigh.view(-1, 81, 2)  # (B * N, 81, 2)
        else:
            child_p = self.children[depth-1]
            nempty = child_p >= 0
            neigh_p = self.neighs[depth-1][nempty]   # (N, 81, 2)
            neigh_p = neigh_p[:, self.lut_parent, :]    # (N, 16, 81, 2)
            child_p = child_p[neigh_p]               # (N, 16, 81, 2)
            invalid = torch.logical_or(child_p < 0, neigh_p < 0)   # (N, 16, 81, 2)
            neigh = child_p * 16 + torch.stack([torch.zeros_like(self.lut_child), self.lut_child], dim=-1)
            neigh[invalid] = -1
            self.neighs[depth] = neigh.view(-1, 81, 2)

    def construct_all_neigh(self):
        r''' A convenient handler for constructing all neighbors.
        '''

        for depth in range(1, self.depth+1):
            self.construct_neigh(depth)

    def search_txyzb(self, query: torch.Tensor, depth: int, nempty: bool = False):
        r''' Searches the hextree nodes given the query points.

        Args:
          query (torch.Tensor): The coordinates of query points with shape
              :obj:`(N, 5)`. The first 4 channels of the coordinates are :obj:`t`, 
              :obj:`x`, :obj:`y`, and :obj:`z`, and the last channel is the batch 
              index. Note that the coordinates must be in range :obj:`[0, 2^depth)`.
          depth (int): The depth of the hextree layer. nemtpy (bool): If true, only
              searches the non-empty hextree nodes.
        '''

        key = txyz2key(query[:, 0], query[:, 1], query[:, 2], query[:, 3], query[:, 4], depth)
        idx = self.search_key(key, depth, nempty)
        return idx

    def search_key(self, query: torch.Tensor, depth: int, nempty: bool = False):
        r''' Searches the hextree nodes given the query points.

        Args:
        query (torch.Tensor): The keys of query points with shape :obj:`(N,)`,
            which are computed from the coordinates of query points.
        depth (int): The depth of the hextree layer. nemtpy (bool): If true, only
            searches the non-empty hextree nodes.
        '''

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

    def get_neigh(self, depth: int, kernel: str = '3333', stride: int = 1,
                  nempty: bool = False):
        r''' Returns the neighborhoods given the depth and a kernel shape.

        Args:
            depth (int): The hextree depth with a value larger than 0 (:obj:`>0`).
            kernel (str): The kernel shape from :obj:`333`, :obj:`311`, :obj:`131`,
                :obj:`113`, :obj:`222`, :obj:`331`, :obj:`133`, and :obj:`313`.
            stride (int): The stride of neighborhoods (:obj:`1` or :obj:`2`). If the
                stride is :obj:`2`, always returns the neighborhood of the first
                siblings.
            nempty (bool): If True, only returns the neighborhoods of the non-empty
                hextree nodes.
        '''

        if stride == 1:
            neigh = self.neighs[depth]
        elif stride == 2:
            # clone neigh to avoid self.neigh[depth] being modified
            neigh = self.neighs[depth][::16].clone()
        else:
            raise ValueError('Unsupported stride {}'.format(stride))

        if nempty:
            child = self.children[depth]
            if stride == 1:
                nempty_node = child >= 0
                neigh = neigh[nempty_node]
            valid = neigh >= 0
            neigh[valid] = child[neigh[valid]].long()  # remap the index

        if kernel == '3333':
            return neigh
        elif kernel in self.lut_kernel:
            lut = self.lut_kernel[kernel]
            return neigh[:, lut]
        else:
            raise ValueError('Unsupported kernel {}'.format(kernel))

    def get_input_feature(self):
        r''' Gets the initial input features.
        '''

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
        r''' Converts averaged points in the hextree to a pointcloud.

        Args:
            rescale (bool): rescale the xyz coordinates to [-1, 1] if True
        ''' 

        depth = self.depth 
        batch_size = self.batch_size

        # by default, use the average points generated when building the hextree
        # from the input point cloud
        txyz = self.points[depth]
        batch_id = self.batch_id(depth, nempty=True)
        
        # txyz is None when the hextree is predicted by a neural network
        if txyz is None:
            t, x, y, z, batch_id = self.txyzb(depth, nempty=True)
            txyz = torch.stack([t, x, y, z], dim=1) + 0.5
        
        # normalize xyz to [-1, 1] since the average points are in range [0, 2 ^ d]
        if rescale:
            scale = 2 ** (1 - depth)
            txyz = self.points[depth].copy()
            txyz[:, 1:] = txyz[:, 1:] * scale - 1.0
        
        # construct Points
        out = Points(txyz, self.normals[depth], self.features[depth],
                     batch_id=batch_id, batch_size=batch_size)
        return out

    def to(self, device: Union[torch.device, str], non_blocking: bool = False):
        r''' Moves the hextree to a specified device.

        Args:
          device (torch.device or str): The destination device.
          non_blocking (bool): If True and the source is in pinned memory, the copy
              will be asynchronous with respect to the host. Otherwise, the argument
              has no effect. Default: False.
        '''

        if isinstance(device, str):
          device = torch.device(device)

        #  If on the save device, directly retrun self
        if self.device == device:
          return self

        def list_to_device(prop):
          return [p.to(device, non_blocking=non_blocking)
                  if isinstance(p, torch.Tensor) else None for p in prop]

        # Construct a new hextree on the specified device
        hextree = Hextree(self.depth, self.full_depth, self.batch_size, device)
        hextree.keys = list_to_device(self.keys)
        hextree.children = list_to_device(self.children)
        hextree.neighs = list_to_device(self.neighs)
        hextree.features = list_to_device(self.features)
        hextree.normals = list_to_device(self.normals)
        hextree.points = list_to_device(self.points)
        hextree.nnum = self.nnum.clone()  # TODO: whether to move nnum to the self.device?
        hextree.nnum_nempty = self.nnum_nempty.clone()
        hextree.batch_nnum = self.batch_nnum.clone()
        hextree.batch_nnum_nempty = self.batch_nnum_nempty.clone()
        return hextree

    def cuda(self, non_blocking: bool = False):
        r''' Moves the hextree to the GPU. '''

        return self.to('cuda', non_blocking)

    def cpu(self):
        r''' Moves the hextree to the CPU. '''

        return self.to('cpu')

    def rng_grid(self, min_, max_):
        r''' Builds a 4D grid in :obj:`[min, max]` (:attr:`max` included).
        '''

        rng = torch.arange(min_, max_+1, dtype=torch.long, device=self.device)
        grid = meshgrid(rng, rng, rng, rng, indexing='ij')
        grid = torch.stack(grid, dim=-1).view(-1, 4)    # ((max_ - min_) ** 4, 4)
        return grid


def merge_hextrees(hextrees: List['Hextree']):
    r''' Merges a list of hextrees into one batch.

    Args:
        hextrees (List[Hextree]): A list of hextrees to merge.
    ''' 

    # init and check 
    hextree = Hextree(depth=hextrees[0].depth, full_depth=hextrees[0].full_depth,
                      batch_size=len(hextrees), device=hextrees[0].device)
    for i in range(1, hextree.batch_size):
        condition = (hextrees[i].depth == hextree.depth and 
                     hextrees[i].full_depth == hextree.full_depth and
                     hextrees[i].device == hextree.device)
        assert condition, 'The check of merge_hextrees failed'
    
    # node num
    batch_nnum = torch.stack(
        [hextrees[i].nnum for i in range(hextrees.batch_size)], dim=1)
    batch_nnum_nempty = torch.stack(
        [hextrees[i].nnum_nempty for i in range(hextrees.batch_size)], dim=1)
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
            key = hextrees[i].keys[d] 

        # children
        children = [None] * hextree.batch_size
        for i in range(hextree.batch_size):
            child = hextrees[i].children[d].clone()  # !! `clone` is used here to avoid
            mask = child >= 0                       # !! modifying the original hextrees
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

    return hextree