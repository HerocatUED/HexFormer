# Convert dataset file to another format
import os
import yaml
import h5py
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


def load_bin(dir_path:str, config_path: str, mode: str, n_frame: int=-1):
    '''
    Load binary file of KITTI.
    
    Args:
    dir_path: path to KITTI.
    config_path: path to KITTI config.
    mode: 'train' use 00-10; 'test' use 11-21.
    n_frame: number of frame to load from single video.
    
    Return:
    points_txyz: list of txyz coordinate with shape n_video * [t_video, n_point=130000, 4]
    depad_num: list of depad num, number of points in corresponding frames.
    semantic: list of corresponding semantic label with shape n_video * [t_video, n_point=13000]
    '''
    print(f"Loading {dir_path}...")
    
    path = dir_path + '/dataset/sequences/'
    videos = os.listdir(path)
    videos.sort()
    if mode == 'train': 
        videos = videos[: 11]
        print('Train, using data sequences', videos)
    else: 
        videos = videos[11:]
        print('Test, using data sequences', videos)
    
    points_txyz = []
    depad_num = []
    for video in videos:
        pcd_dir = path + video + '/velodyne/'
        pcd_files = os.listdir(pcd_dir)
        pcd_files.sort()
        pcd_files = pcd_files[: n_frame]
        pcd = np.zeros((len(pcd_files), 130000, 4)) # pad every frame to 13000 points
        depad = np.zeros(len(pcd_files), dtype=int) # depad num, number of points in this frame
        for i in tqdm(range(len(pcd_files)), desc=f'points {video}'):
            scan = np.fromfile(pcd_dir + pcd_files[i], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            depad[i] = np.shape(scan)[0]
            # put in attribute
            points = scan[:, 0:3]    # get xyz
            remissions = scan[:, 3]  # get remission
            pcd[i, :depad[i], 0] = i
            pcd[i, :depad[i], 1:] = points
        points_txyz.append(pcd)
        depad_num.append(depad)
    
    if mode != 'train': return points_txyz, depad_num
    
    semantic = []
    for k, video in enumerate(videos):
        label_dir = path + video + '/labels/'
        label_files = os.listdir(label_dir)
        label_files.sort()
        label_files = label_files[: n_frame]
        labels = np.zeros((len(label_files), 130000), dtype=int) # pad every frame to 13000 points
        for i in tqdm(range(len(label_files)), desc=f'label {video}'):  
            label = np.fromfile(label_dir + label_files[i], dtype=np.uint32)
            label = label.reshape((-1))
            # only fill in attribute if the right size
            if label.shape[0] == depad_num[k][i]:
                sem_label = label & 0xFFFF  # semantic label in lower half
                inst_label = label >> 16    # instance id in upper half
            else:
                print("Points shape: ", points.shape)
                print("Label shape: ", label.shape)
                raise ValueError("Scan and Label don't contain same number of points")
            # sanity check
            assert((sem_label + (inst_label << 16) == label).all())
            
            labels[i, :depad_num[k][i]] = sem_label
        labels = remap(labels, config_path, False)
        semantic.append(labels)      
    
    return points_txyz, semantic, depad_num
    
    
def construct_dataset(points_txyz, semantic, dataset_name: str, clip_length: int, depad_num: list, val_videos: list):
    '''
    Construct dataset and config files.
    TODO support test dataset construction
    Args:
    points_txyz: array/list of txyz coordinate with shape [n_video, t_video, n_point, 4]
    semantic: array/list of corresponding semantic label with shape [n_video, t_video, n_point]
    dataset_name: name of folder to save clipped file, $batch size$ will be number of clipped file to read in a batch.
    clip_length: number of frame in a single clip.
    depad_num: if is not None, only take first depad[i] points in frame i.
    val_videos: if is not None, videos that (id) in val_videos will be taken as validation data. 
    '''  
    print(f"Constructing dataset into {dataset_name}...")
    assert len(points_txyz) == len(semantic), 'semantic should be corresponding to points_txyz'
    
    os.makedirs(dataset_name, exist_ok=True)
    
    train_list = ''
    val_list = ''

    for v in range(len(points_txyz)): # loop over videos
        t_video, _, _ = points_txyz[v].shape
        N = t_video // clip_length
    
        # depad
        pcd = [None] * N
        label = [None] * N
        for j in tqdm(range(N), desc=f'depad {v}'):
            index = np.cumsum(depad_num[v][j*clip_length:(j+1)*clip_length])
            index = np.pad(index, (1,0), 'constant', constant_values=0)
            pcd[j] = np.zeros((index[-1], 4))
            label[j] = np.zeros((index[-1]))
            for k in range(clip_length):
                id = j*clip_length + k
                assert index[k+1]-index[k] == depad_num[v][id]
                pcd[j][index[k]:index[k+1], :] = points_txyz[v][id][:depad_num[v][id]]
                label[j][index[k]:index[k+1]] = semantic[v][id][:depad_num[v][id]]  

        for i in range(N):
            np.savez(f'{dataset_name}/data_{v}_{i}.npz', points=pcd[i], labels=label[i])
            if v in val_videos: val_list += f'{dataset_name}/data_{v}_{i}.npz\n'
            else: train_list += f'{dataset_name}/data_{v}_{i}.npz\n'
        
        # left frames
        if t_video % clip_length > 0:
            
            index = np.cumsum(depad_num[v][N*clip_length:])
            index = np.pad(index, (1,0), 'constant', constant_values=0)
            pcd = np.zeros((index[-1], 4))
            label = np.zeros((index[-1]))
            for k in range(t_video % clip_length):
                id = N*clip_length + k
                pcd[index[k]:index[k+1], :] = points_txyz[v][id][:depad_num[v][id]]
                label[index[k]:index[k+1]] = semantic[v][id][:depad_num[v][id]] 
                 
            np.savez(f'{dataset_name}/data_{v}_{N}.npz', points=pcd, labels=label)
            if val_videos is not None and v in val_videos: val_list += f'{dataset_name}/data_{v}_{N}.npz\n'
            else: train_list += f'{dataset_name}/data_{v}_{N}.npz\n'
    # save config file
    f_train = open(f'{dataset_name}/train_data.txt', 'w')
    f_val = open(f'{dataset_name}/val_data.txt', 'w')
    f_train.write(train_list)
    f_train.close()
    f_val.write(val_list)
    f_val.close()



if __name__ == '__main__':
    
    # example of building dataset using KITTI
    clip_length = 4
    kitti_dir = '/mnt/sdc/wangrh/data/SemanticKITTI'
    dataset_name = f'/mnt/sdc/wangx/HexFormer/dataset/kitti/frame{clip_length}'
    config_path = '/mnt/sdc/wangx/HexFormer/data_utils/config/semantic-kitti.yaml'
    points_txyz, semantic, depad_num = load_bin(kitti_dir, config_path, dataset_name, 'train', 100, True)
    construct_dataset(points_txyz, semantic, dataset_name, clip_length, depad_num, val_videos=[8])


    