# Convert a double-precision dataset to single-precision
# Written by Xiang Wang
import torch
import h5py
import numpy as np
from tqdm import tqdm


def float64to32():
    chunk_size = 30

    for filename in ['train1', 'train2', 'train3', 'train4']:
        with h5py.File('/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5'+'/'+filename+'.h5', 'r') as f:
            with h5py.File('/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5'+'/'+filename+'_float32.h5', 'w') as new_f:
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


def h52npy(n_video):
    with h5py.File('/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5'+'/train1_float32.h5', 'r') as f:
        for dataset_name in ['semantic', 'pcd']:
            print(dataset_name)
            data = np.array(f[dataset_name][:n_video, :160])
            print(f'{dataset_name}', np.shape(data))
            np.save(f'./dataset/{dataset_name}', data)


def xyz2txyz(clip_length: int = 5):
    points_xyz = np.load('./dataset/pcd.npy')
    semantic = np.load('./dataset/semantic.npy')    

    # Translate data to txyz format
    n_video, t_video, n_point, _ = points_xyz.shape
    points_txyz = np.concatenate([np.zeros([n_video, t_video, n_point, 1]), points_xyz], axis=-1)
    points_txyz[:, :, :, 0] += (np.arange(t_video) + 1)[None, :, None]

    clip_rate = int(t_video / clip_length)
    N = clip_rate * n_video
    # print(clip_rate, n_video)
    points_txyz = torch.Tensor(points_txyz.reshape([N, -1, 4]))  
    semantic = torch.Tensor(semantic.reshape([N, -1]))           

    f_train = open('./config/train_data.txt', 'w')
    f_val = open('./config/val_data.txt', 'w')
    train_list = ''
    val_list = ''
    for i in range(N):
        np.savez(f'./dataset/train_{i}.npz',
                 points=points_txyz[i], labels=semantic[i],)
        if i % 5 == 0:
            train_list += f'./dataset/train_{i}.npz\n'
        else:
            val_list += f'./dataset/train_{i}.npz\n'
    f_train.write(train_list)
    f_train.close()
    f_val.write(val_list)
    f_val.close()
    
def trans_visualize(rand_ids, logdir):
    for i, rand_id in enumerate(rand_ids):
        points = np.load(f"../logs/log_{logdir}_hoi4d/result_sample/points_{rand_id}.npz")
        points = np.array(points['arr_0'][:8192, 1:])
        print("points", np.shape(points))
        pred = np.load(f"../logs/log_{logdir}_hoi4d/result_sample/pred_{rand_id}.npz")
        pred = np.array(pred['arr_0'][:8192])
        print("pred", np.shape(pred))
        label = np.load(f"../logs/log_{logdir}_hoi4d/result_sample/label_{rand_id}.npz")
        label = np.array(label['arr_0'][:8192])
        print("label", np.shape(label))
        prediction = np.concatenate([points, np.expand_dims(pred, 1)], axis=-1)
        groundtruth = np.concatenate([points, np.expand_dims(label, 1)], axis=-1)
        np.save(f"./visualize/prediction_{i+2}.npy", prediction)
        np.save(f"./visualize/groundtruth_{i+2}.npy", groundtruth)


if __name__ == '__main__':
    # float_64to32()
    
    # rand_ids = [0.07, 0.09]
    # logdir = 'SingleVideo'
    # trans_visualize(rand_ids, logdir)
    
    # h52npy(50)
    xyz2txyz(8)
