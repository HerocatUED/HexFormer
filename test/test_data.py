import numpy as np
import torch 
import os 
import sys 
sys.path.append('..')
import hextree.shuffled_key as sk
import hextree as hextree
from tqdm import tqdm
import time
from typing import Union
from itertools import product


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_xyz(filename: str, xyz: np.ndarray, color: Union[str, np.ndarray, None] = 'random'):
    if isinstance(color, str):
        if color == 'random':
            color = np.random.randint(low=0, high=255, size=(1, 3))
        else:
            raise
    if not filename.endswith('.txt'):
        filename = filename + '.txt'
    with open(filename, 'w') as f:
        if color is None:
            f.write(str(xyz.tolist()).replace('[', '').replace('],', ',\n').replace(']', '').replace(',', ''))
        elif color.shape == (1, 3) or color.shape == (3, ):
            color = color.reshape(1, 3)
            xyz = np.concatenate([xyz, color.repeat(len(xyz), axis=0)], axis=-1)
            f.write(str(xyz.tolist()).replace('[', '').replace('],', ',\n').replace(']', '').replace(',', ''))
        elif color.shape == xyz.shape:
            xyz = np.concatenate([xyz, color], axis=1)
            f.write(str(xyz.tolist()).replace('[', '').replace('],', ',\n').replace(']', '').replace(',', ''))
        else:
            raise

def read_data(save_raw=True, **kwargs):
    # Read and save raw data
    points_xyz = np.load('../data/points.npy')     # (N, 300, 8192, 3)
    semantic = np.load('../data/semantic.npy')     # (N, 300, 8192)

    if save_raw:
        if not os.path.exists('../data/raw'):
            os.mkdir('../data/raw')
        for v, frs in zip(kwargs['video_id'], kwargs['frame_id']):
            for fr in frs:
                mincoord = points_xyz[v].min(axis=(0, 1))
                maxcoord = points_xyz[v].max(axis=(0, 1))
                center = (mincoord + maxcoord) * 0.5
                box_size = (maxcoord - mincoord).max() + 1.0e-6
                save_xyz(f'../data/raw/raw{v}_frame{fr+1}', (points_xyz[v, fr] - center) / box_size + 0.5, color=np.array([252, 233, 79]))  

    # Translate data to txyz format
    n_video, t_video, n_point, _ = points_xyz.shape
    points_txyz = np.concatenate([np.zeros([n_video, t_video, n_point, 1]), points_xyz], axis=-1)
    points_txyz[:, :, :, 0] += (np.arange(t_video) + 1)[None, :, None]

    points_txyz = torch.Tensor(points_txyz.reshape([n_video, -1, 4]))   # (N, 2457600, 4)
    semantic = torch.Tensor(semantic.reshape([n_video, -1]))            # (N, 2457600)

    # Normalization of points
    pcds = [hextree.Points(points=points_txyz[i], labels=semantic[i]) for i in range(n_video)]
    for pcd in pcds:
        bbmin, bbmax = pcd.bbox_xyz()
        pcd.normalize_xyz(bbmin, bbmax)
    
    return pcds

def timing(pcd, depth=7, frame=16):
    # Testing and timing hextree building process
    point_num = 8192 * frame 
    points = pcd[:point_num].to(device)
    htree = hextree.Hextree(depth=depth).to(device)

    start_time = time.time()
    htree.build_hextree(points)
    mid_time = time.time()
    htree.construct_all_neigh()
    end_time = time.time()
    print(f'Depth: {depth}, \tframe: {frame}, \thextree building time: {(mid_time - start_time)}, \tneighbour constructing time: {(end_time - mid_time)}')

def save_pcd(pcd, depth_testset=[5, 6, 7, 8, 9, 10], frame=8, color_mode: str='test_neigh'):
    # Save pointcloud decoded from hextree

    if not os.path.exists('../data/visualization'):
        os.mkdir('../data/visualization')
    point_num = frame * 8192
    for depth in depth_testset:
        htree = hextree.Hextree(depth=depth).to(device)
        points = pcd[:point_num].to(device)
        htree.build_hextree(points)
        htree.construct_all_neigh()
        txyz = torch.stack(sk.key2txyz(htree.keys[-1][htree.children[-1] >= 0], depth=depth), dim=1)[:, :4].numpy()
        center_point = np.random.randint(low=0, high=len(txyz), size=10)
        neighs = htree.get_neigh(depth=depth, kernel='3333', nempty=True)[center_point, :, 1]
        neighs = neighs[neighs >= 0]
        t = txyz[:, 0]

        for fr in range(1, frame+1):
            idx = np.where(t == fr)
            xyz = txyz[idx][:, 1:] / (1 << depth)
            
            # Test neighbour searching
            if color_mode == 'test_neigh':
                np.random.seed(41)
                colors = np.random.randint(low=0, high=255, size=(2, 3))
                color = np.zeros((len(txyz), 3))
                color[:, :] = colors[0]
                center_point = np.random.randint(low=0, high=len(xyz), size=8)
                neighs = htree.get_neigh(depth=depth, kernel='3333', nempty=True)[idx][center_point, :, 1]
                neighs = neighs[neighs >= 0]
                color[neighs] = colors[1]
                color = color[idx]
            elif color_mode == 'test_tree':
                n_zones = 8
                np.random.seed(41)
                colors = np.random.randint(low=0, high=255, size=(n_zones, 3))
                color = np.zeros((len(txyz), 3))
                zone_len = len(txyz) // n_zones + 1
                for i in range(n_zones):
                    color[zone_len * i: zone_len * (i+1), :] = colors[i]
                color = color[idx]
            else:
                color = None

            filename = f'res_depth{depth}_frame{fr}'
            if color_mode.startswith('test_'):
                filename += '_' + color_mode[5:]
            save_xyz(f'../data/visualization/{filename}', xyz, color)

if __name__ == '__main__':
    pcds = read_data(save_raw=True, video_id=[0], frame_id=[[0, 1, 2, 3, 4, 5, 6, 7]])
    # pcds = read_data(save_raw=False)
    # timing(pcds[0])
    save_pcd(pcds[0], color_mode='test_tree')