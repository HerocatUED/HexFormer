# Convert dataset file to another format

import os
import yaml
import numpy as np
from tqdm import tqdm


def remap(semantic: np.array, config_path: str, inverse: bool = False):
    '''
    Remap semantic classes.
    
    Args:
    semantic: semantic classes to remap.
    config_path: path to KITTI config.
    inverse: class2num if True, num2class if False. NOTE: See KITTI config for more.
    '''
    DATA = yaml.safe_load(open(config_path, 'r'))

    # get number of interest classes, and the label mappings
    if inverse:
        print("Mapping xentropy to original labels")
        remapdict = DATA["learning_map_inv"]
    else:
        remapdict = DATA["learning_map"]

    # make lookup table for mapping
    maxkey = max(remapdict.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    return remap_lut[semantic]
    
    
def construct_train(dir_path:str, config_path: str, dataset_name: str, clip_length: int, val_videos: list, n_frame:int = -1):
    '''
    Construct dataset and config files.
    Args:
    dir_path: path to KITTI.
    config_path: path to KITTI config.
    dataset_name: name of folder to save clipped file, $batch size$ will be number of clipped file to read in a batch.
    clip_length: number of frame in a single clip.
    val_videos: if is not None, videos that (id) in val_videos will be taken as validation data. 
    n_frame: number of frame to load from single video.
    '''  
    train_list = ''
    val_list = ''

    path = dir_path + '/dataset/sequences/'
    videos = os.listdir(path)
    videos.sort()
    videos = videos[: 11]
    print('Train, using data sequences', videos)

    for video in videos:
        # point clouds xyz2txyz
        pcd_dir = path + video + '/velodyne/'
        pcd_files = os.listdir(pcd_dir)
        pcd_files.sort()
        pcd_files = pcd_files[: min(n_frame, len(pcd_files))]
        points_txyz = np.zeros((len(pcd_files), 130000, 4)) # pad every frame to 13000 points
        depad = np.zeros(len(pcd_files), dtype=int) # depad num, number of points in this frame
        for i in tqdm(range(len(pcd_files)), desc=f'points {video}'):
            scan = np.fromfile(pcd_dir + pcd_files[i], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            depad[i] = np.shape(scan)[0]
            # put in attribute
            points = scan[:, 0:3]    # get xyz
            remissions = scan[:, 3]  # get remission
            points_txyz[i, :depad[i], 0] = i
            points_txyz[i, :depad[i], 1:] = points
    
        # labels
        label_dir = path + video + '/labels/'
        label_files = os.listdir(label_dir)
        label_files.sort()
        label_files = label_files[: min(n_frame, len(label_files))]
        labels = np.zeros((len(label_files), 130000), dtype=int) # pad every frame to 13000 points
        for i in tqdm(range(len(label_files)), desc=f'label {video}'):  
            label = np.fromfile(label_dir + label_files[i], dtype=np.uint32)
            label = label.reshape((-1))
            # only fill in attribute if the right size
            if label.shape[0] == depad[i]:
                sem_label = label & 0xFFFF  # semantic label in lower half
                inst_label = label >> 16    # instance id in upper half
            else:
                print("Points shape: ", points.shape)
                print("Label shape: ", label.shape)
                raise ValueError("Scan and Label don't contain same number of points")
            # sanity check
            assert((sem_label + (inst_label << 16) == label).all())
            
            labels[i, :depad[i]] = sem_label
        labels = remap(labels, config_path, False)      

        # construct dataset
        t_video, _, _ = points_txyz.shape
        N = t_video // clip_length
        pcd = [None] * N
        label = [None] * N
        for i in range(N):
            index = np.cumsum(depad[i*clip_length:(i+1)*clip_length])
            index = np.pad(index, (1,0), 'constant', constant_values=0)
            pcd[i] = np.zeros((index[-1], 4))
            label[i] = np.zeros((index[-1]))
            for k in range(clip_length):
                id = i*clip_length + k
                assert index[k+1]-index[k] == depad[id]
                pcd[i][index[k]:index[k+1], :] = points_txyz[id][:depad[id]]
                label[i][index[k]:index[k+1]] = labels[id][:depad[id]]  
            np.savez(f'{dataset_name}/data_{video}_{i}.npz', points=pcd[i], labels=label[i])
            if video in val_videos: val_list += f'{dataset_name}/data_{video}_{i}.npz\n'
            else: train_list += f'{dataset_name}/data_{video}_{i}.npz\n'
        
        # left frames
        if t_video % clip_length > 0:
            index = np.cumsum(depad[N*clip_length:])
            index = np.pad(index, (1,0), 'constant', constant_values=0)
            pcd = np.zeros((index[-1], 4))
            label = np.zeros((index[-1]))
            for k in range(t_video % clip_length):
                id = N*clip_length + k
                pcd[index[k]:index[k+1], :] = points_txyz[id][:depad[id]]
                label[index[k]:index[k+1]] = labels[id][:depad[id]] 
            np.savez(f'{dataset_name}/data_{video}_{N}.npz', points=pcd, labels=label)
            if video in val_videos: val_list += f'{dataset_name}/data_{video}_{N}.npz\n'
            else: train_list += f'{dataset_name}/data_{video}_{N}.npz\n'
            
    # save config file
    f_train = open(f'{dataset_name}/train_data.txt', 'w')
    f_val = open(f'{dataset_name}/val_data.txt', 'w')
    f_train.write(train_list)
    f_train.close()
    f_val.write(val_list)
    f_val.close()
    

def construct_test(dir_path:str, dataset_name: str, clip_length: int):
    '''
    Construct dataset and config files.
    Args:
    dataset_name: name of folder to save clipped file, $batch size$ will be number of clipped file to read in a batch.
    clip_length: number of frame in a single clip.
    '''  
    test_list = ''

    path = dir_path + '/dataset/sequences/'
    videos = os.listdir(path)
    videos.sort()
    videos = videos[11:]
    print('Test, using data sequences', videos)
    
    for video in videos:
        # point clouds xyz2txyz
        pcd_dir = path + video + '/velodyne/'
        pcd_files = os.listdir(pcd_dir)
        pcd_files.sort()
        points_txyz = np.zeros((len(pcd_files), 130000, 4)) # pad every frame to 13000 points
        depad = np.zeros(len(pcd_files), dtype=int) # depad num, number of points in this frame
        for i in tqdm(range(len(pcd_files)), desc=f'points {video}'):
            scan = np.fromfile(pcd_dir + pcd_files[i], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            depad[i] = np.shape(scan)[0]
            # put in attribute
            points = scan[:, 0:3]    # get xyz
            remissions = scan[:, 3]  # get remission
            points_txyz[i, :depad[i], 0] = i
            points_txyz[i, :depad[i], 1:] = points

        # construct dataset
        t_video, _, _ = points_txyz.shape
        N = t_video // clip_length
        pcd = [None] * N
        for i in range(N):
            index = np.cumsum(depad[i*clip_length:(i+1)*clip_length])
            index = np.pad(index, (1,0), 'constant', constant_values=0)
            pcd[i] = np.zeros((index[-1], 4))
            for k in range(clip_length):
                id = i*clip_length + k
                assert index[k+1]-index[k] == depad[id]
                pcd[i][index[k]:index[k+1], :] = points_txyz[id][:depad[id]] 
            np.savez(f'{dataset_name}/data_{video}_{i}.npz', points=pcd[i], labels=None)
            test_list += f'{dataset_name}/data_{video}_{i}.npz\n'

        # left frames
        if t_video % clip_length > 0:
            index = np.cumsum(depad[N*clip_length:])
            index = np.pad(index, (1,0), 'constant', constant_values=0)
            pcd = np.zeros((index[-1], 4))
            for k in range(t_video % clip_length):
                id = N*clip_length + k
                pcd[index[k]:index[k+1], :] = points_txyz[id][:depad[id]]
            np.savez(f'{dataset_name}/data_{video}_{N}.npz', points=pcd, labels=None)
            test_list += f'{dataset_name}/data_{video}_{N}.npz\n'
            
    # save config file
    f_test = open(f'{dataset_name}/train_data.txt', 'w')
    f_test.write(test_list)
    f_test.close()


def construct_dataset(dir_path:str, config_path: str, dataset_name: str, clip_length: int, val_videos: list, n_frame:int = -1):
    '''
    Construct dataset and config files.
    mode: 'train' use 00-10; 'test' use 11-21.
    
    Args:
    dir_path: path to KITTI.
    config_path: path to KITTI config.
    dataset_name: name of folder to save clipped file, $batch size$ will be number of clipped file to read in a batch.
    clip_length: number of frame in a single clip.
    val_videos: if is not None, videos that (id) in val_videos will be taken as validation data. 
    n_frame: number of frame to load from single video.
    '''  
    print(f"Constructing dataset into {dataset_name}...")
    os.makedirs(dataset_name, exist_ok=True)
    print(f"Constructing trainingset into {dataset_name}...")
    construct_train(dir_path, config_path, dataset_name, clip_length, val_videos, n_frame)
    print(f"Constructing testset into {dataset_name}...")
    construct_test(dir_path, dataset_name, clip_length)


if __name__ == '__main__':
    
    # example of building dataset using KITTI
    clip_length = 4
    kitti_dir = '/mnt/sdc/wangrh/data/SemanticKITTI'
    dataset_name = f'/mnt/sdc/wangx/HexFormer/dataset/kitti/frame{clip_length}_full'
    config_path = '/mnt/sdc/wangx/HexFormer/data_utils/config/semantic-kitti.yaml'
    construct_dataset(kitti_dir, config_path, dataset_name, clip_length, val_videos=[8])


    