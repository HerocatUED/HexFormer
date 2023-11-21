import torch 
import sys 
sys.path.append('..')
import hextree as hextree
import time
from utils import get_points

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def timing(pcd, depth=10, n_frame=16, need_neigh=False):
    # Testing and timing hextree building process
    point_num = 8192 * n_frame 
    points = pcd[:point_num].to(device)
    htree = hextree.Hextree(depth=depth).to(device)

    start_time = time.time()
    htree.build_hextree(points)
    mid_time = time.time()
    if need_neigh:
        htree.construct_all_neigh()
    end_time = time.time()
    print(f'Depth: {depth}, \tframe: {n_frame}, \thextree building time: {(mid_time - start_time)}' + 
          (f', \tneighbour constructing time: {(end_time - mid_time)}' if need_neigh else ''))

def test_timing(data_path, label_path, video_id, min_depth=5, max_depth=14, n_frame=16, need_neigh=False):
    pcd = get_points(data_path, label_path, video_id=video_id, n_frame=n_frame)
    for depth in range(min_depth, max_depth+1):
        if (1 << depth) >= n_frame:
            timing(pcd, depth, n_frame, need_neigh)

'../data/semantic.npy'
if __name__ == '__main__':
    test_timing(data_path='../data/points.npy', label_path='../data/semantic.npy', video_id=0)