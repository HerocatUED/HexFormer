import os
import torch
import numpy as np
import sys
sys.path.append('..')
from hextree import Points, Hextree, merge_hextrees
from typing import Union, Optional


def points_3Dto4D(points_xyz: np.ndarray):
    n_video, t_video, n_point, _ = points_xyz.shape
    points_txyz = np.concatenate([np.zeros([n_video, t_video, n_point, 1]), points_xyz], axis=-1)
    points_txyz[:, :, :, 0] += (np.arange(t_video) + 1)[None, :, None]

    points_txyz = torch.Tensor(points_txyz.reshape([n_video, -1, 4]))   # (n_video, t_video * n_point, 4)
    return points_txyz

def read_points(
        data_path: str,
        label_path: Optional[str] = None):
    # Read and save raw data
    points_xyz = np.load(data_path)     # (N, 300, 8192, 3)
    semantic = None
    if label_path is not None:
        semantic = np.load(label_path)

    # Translate data to txyz format
    n_video, _, _, _ = points_xyz.shape
    points_txyz = points_3Dto4D(points_xyz)
    if semantic is not None:
        semantic = torch.Tensor(semantic.reshape([n_video, -1]))            # (N, 2457600)

    # Normalization of points
    pcds = [Points(points=points_txyz[i], 
                   labels=semantic[i] if semantic is not None else None) for i in range(n_video)]
    for pcd in pcds:
        pcd.normalize_xyz()
    return pcds

def get_points(data_path, label_path, video_id, n_frame=8):
    points = read_points(data_path=data_path, label_path=label_path)
    points = points[video_id][:n_frame*8192]
    return points

def get_hextree(data_path, label_path, video_id, n_frame=8, 
                depth=8, full_depth=1, device='cpu', need_neigh=False):
    point_cloud = get_points(data_path, label_path, video_id, n_frame)
    hextree = Hextree(depth=depth,full_depth=full_depth, device=device)
    hextree.build_hextree(point_cloud)
    if need_neigh:
        hextree.construct_all_neigh()
    return hextree

def get_batch_hextree(data_path, label_path, video_ids=[0, 1], n_frame=8, 
                      depth=8, full_depth=1, device='cpu', need_neigh=False):
    hextrees = []
    for video_id in video_ids:
        htree_i = get_hextree(video_id, n_frame, depth, full_depth, device, need_neigh=False)
        hextrees.append(htree_i)
    hextree = merge_hextrees(hextrees)
    if need_neigh:
        hextree.construct_all_neigh()
    return hextree