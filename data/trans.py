# Convert a double-precision dataset to single-precision
# Written by Xiang Wang
import torch
import h5py
import numpy as np
from tqdm import tqdm


def float_64to32():
    chunk_size = 30

    for filename in ['train1', 'train2', 'train3', 'train4']:
        with h5py.File('../HOI4D_dataset/seg_data_h5'+'/'+filename+'.h5', 'r') as f:
            with h5py.File('../HOI4D_dataset/seg_data_h5'+'/'+filename+'_float32.h5', 'w') as new_f:
                for dataset_name in ['center', 'semantic', 'pcd']:

                    print(dataset_name)

                    original_data = f[dataset_name]
                    print(type(original_data[0].dtype))
                    shape = original_data.shape

                    if dataset_name == 'semantic':
                        new_f.create_dataset(
                            dataset_name, shape=shape, dtype=np.int8, chunks=True)
                    else:
                        new_f.create_dataset(
                            dataset_name, shape=shape, dtype=np.float32, chunks=True)

                    total_data = shape[0]
                    num_iterations = total_data // chunk_size

                    for i in tqdm(range(num_iterations)):
                        start_idx = i * chunk_size
                        end_idx = (i + 1) * chunk_size

                        chunk_data = original_data[start_idx:end_idx]

                        if dataset_name == 'semantic':
                            single_precision_chunk = chunk_data.astype(np.int8)
                        else:
                            single_precision_chunk = chunk_data.astype(
                                np.float32)

                        new_f[dataset_name][start_idx:end_idx] = single_precision_chunk

                    if total_data % chunk_size != 0:
                        start_idx = num_iterations * chunk_size
                        end_idx = total_data

                        chunk_data = original_data[start_idx:end_idx]
                        single_precision_chunk = chunk_data.astype(np.float32)

                        new_f[dataset_name][start_idx:end_idx] = single_precision_chunk

    print("Done.")


def trans(clip_length: int = 5):
    points_xyz = np.load('points.npy')     # (N_video, 300, 8192, 3)
    semantic = np.load('semantic.npy')     # (N_video, 300, 8192)

    # Translate data to txyz format
    n_video, t_video, n_point, _ = points_xyz.shape
    points_txyz = np.concatenate([np.zeros([n_video, t_video, n_point, 1]), points_xyz], axis=-1)
    points_txyz[:, :, :, 0] += (np.arange(t_video) + 1)[None, :, None]

    clip_rate = int(t_video / clip_length)
    N = clip_rate * n_video
    points_txyz = torch.Tensor(points_txyz.reshape([N, -1, 4]))   # (N, 2457600, 4)
    semantic = torch.Tensor(semantic.reshape([N, -1]))            # (N, 2457600)

    f = open('./dataset/train_npz.txt', 'w')
    file_list = ''
    for i in range(N):
        np.savez(f'./dataset/train_{i}.npz',
                 points=points_txyz[i], labels=semantic[i],)
        file_list += f'./dataset/train_{i}.npz\n'
    f.write(file_list)
    f.close()


if __name__ == '__main__':
    # float_64to32()
    trans()
    # x = np.load('dataset/train_0.npz')
    # print(np.shape(x['points']))
