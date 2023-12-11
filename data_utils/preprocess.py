# Convert dataset file to another format
import os
import torch
import h5py
import numpy as np
from tqdm import tqdm

def h5float64to32(path: str, chunk_size: int = 30):
    '''
    convert h5 files from float64 to float32 to save memory
    
    Args:
    chunk_size: size of data processed at one time NOTE set it according to memory of your device
    path: path to dir containing h5 files of HOI4D
    '''
    print("Converting to float32...")
    
    for filename in ['train1', 'train2', 'train3', 'train4']:
        with h5py.File(path+'/'+filename+'.h5', 'r') as f:
            with h5py.File(path+'/'+filename+'_float32.h5', 'w') as new_f:
                for dataset_name in ['center', 'semantic', 'pcd']:
                    
                    original_data = f[dataset_name]
                    print(dataset_name, type(original_data[0].dtype))
                    shape = original_data.shape

                    if dataset_name == 'semantic':
                        new_f.create_dataset(dataset_name, shape=shape, dtype=np.int8, chunks=True)
                    else:
                        new_f.create_dataset(dataset_name, shape=shape, dtype=np.float32, chunks=True)

                    total_data = shape[0]
                    num_iterations = total_data // chunk_size

                    for i in tqdm(range(num_iterations)):
                        start_idx = i * chunk_size
                        end_idx = (i + 1) * chunk_size
                        chunk_data = original_data[start_idx:end_idx]
                        if dataset_name == 'semantic':
                            single_precision_chunk = chunk_data.astype(np.int8)
                        else:
                            single_precision_chunk = chunk_data.astype(np.float32)
                        new_f[dataset_name][start_idx:end_idx] = single_precision_chunk
                        
                    if total_data % chunk_size != 0:
                        start_idx = num_iterations * chunk_size
                        end_idx = total_data
                        chunk_data = original_data[start_idx:end_idx]
                        single_precision_chunk = chunk_data.astype(np.float32)
                        new_f[dataset_name][start_idx:end_idx] = single_precision_chunk
    print("Done.")


def load_h5(h5path:str, n_video: int, n_frame: int, dataset_name: str, save:bool=False):
    '''
    load h5 file of HOI4D
    
    Args:
    h5path: path of h5 file to load
    n_video: number of video used in single h5 file
    n_frame: number of frame used in single video 
    save: save to file or not
    dataset_name: name of folder to save npy file, only used if $save$ is True.
    
    Return:
    pcd: array of xyz coordinate with shape [n_video, t_video, n_point, 3]
    semantic: array of corresponding semantic label with shape [n_video, t_video, n_point, 1]
    '''
    print(f"Loading {h5path}...")
    
    pcd = None
    semantic = None
    with h5py.File(h5path, 'r') as f:
        pcd = np.array(f['pcd'][:n_video, :n_frame])
        semantic = np.array(f['semantic'][:n_video, :n_frame])
        print('pcd', np.shape(pcd))
        print('semantic', np.shape(semantic))
        if save: 
            os.mkdir(dataset_name, exist_ok=True)
            np.save(f'{dataset_name}/pcd', pcd)
            np.save(f'{dataset_name}/semantic', semantic)
    return pcd, semantic


def load_bin():
    pass


def xyz2txyz(points_xyz: np.array):
    '''
    convert xyz array to txyz array
    
    Args:
    points_xyz: array of xyz coordinate with shape [n_video, t_video, n_point, 3]
    
    Return:
    points_txyz [n_video, t_video, n_point, 4]
    '''
    print("Converting xyz to txyz")
    
    n_video, t_video, n_point, _ = points_xyz.shape
    print(f'video num: {n_video}')
    print(f'frames per video: {t_video}')
    points_txyz = np.concatenate([np.zeros([n_video, t_video, n_point, 1]), points_xyz], axis=-1)
    points_txyz[:, :, :, 0] += (np.arange(t_video) + 1)[None, :, None]
    return points_txyz
    
    
def construct_dataset(points_txyz: np.array, semantic: np.array, dataset_name: str, config_name: str, clip_length: int):
    '''
    convert xyz array to txyz array and construct dataset config files
    
    Args:
    points_txyz: array of xyz coordinate with shape [n_video, t_video, n_point, 4]
    semantic: array of corresponding semantic label with shape [n_video, t_video, n_point, 1]
    dataset_name: name of folder to save clipped file, $batch size$ will be number of clipped file to read in a batch 
    config_name: name of config folder to save train & validation split config
    clip_length: number of frame in a single clip
    '''  
    print("Constructing dataset...")
    os.mkdir(dataset_name, exist_ok=True)
    os.mkdir(config_name, exist_ok=True)
    
    n_video, t_video, _, _ = points_txyz.shape
    clip_rate = int(t_video / clip_length)
    N = clip_rate * n_video
    
    points_txyz = torch.Tensor(points_txyz.reshape([N, -1, 4]))  
    semantic = torch.Tensor(semantic.reshape([N, -1]))           

    train_list = ''
    val_list = ''

    for i in tqdm(range(N)):
        np.savez(f'{dataset_name}/data_{i}.npz', points=points_txyz[i], labels=semantic[i],)
        if i % 5 == 0:
            val_list += f'{dataset_name}/data_{i}.npz\n'
        else:
            train_list += f'{dataset_name}/data_{i}.npz\n'
    
    f_train = open(f'{config_name}/train_data.txt', 'w')
    f_val = open(f'{config_name}/val_data.txt', 'w')
    f_train.write(train_list)
    f_train.close()
    f_val.write(val_list)
    f_val.close()



if __name__ == '__main__':
    
    # example of building dataset using HOI4D
    clip_length = 8
    h5_dir = '/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5'
    dataset_name = f'/mnt/sdc/wangx/HexFormer/dataset/hoi4d/frame{clip_length}'
    config_name = '/mnt/sdc/wangx/HexFormer/data_utils/config/hoi4d'
    # h5float64to32(h5_dir, 100) NOTE run only if you need to save memory
    pcd, semantic = load_h5(h5_dir+'/train1_float32.h5', 500, 160, dataset_name, True)
    points_txyz = xyz2txyz(pcd)
    construct_dataset(points_txyz, semantic, dataset_name, config_name, clip_length)
    
    # example of building dataset using KITTI
    # clip_length = 8
    # dataset_name = f'/mnt/sdc/wangx/HexFormer/dataset/kitti/frame{clip_length}'
    # config_name = '/mnt/sdc/wangx/HexFormer/data_utils/config/kitti'
    # pcd, semantic = load_bin() #TODO
    # points_txyz = xyz2txyz(pcd)
    # construct_dataset(points_txyz, semantic, dataset_name, config_name, clip_length)
    