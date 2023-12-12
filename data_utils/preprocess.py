# Convert dataset file to another format
import os
import yaml
import h5py
import numpy as np
from tqdm import tqdm

def h5float64to32(path: str, chunk_size: int = 30):
    '''
    Convert h5 files from float64 to float32 to save memory.
    
    Args:
    chunk_size: size of data processed at one time. NOTE set it according to memory of your device.
    path: path to dir containing h5 files of HOI4D.
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


def load_h5(h5path:str, dataset_name: str, n_video: int=-1, n_frame: int=-1, save:bool=False):
    '''
    Load h5 file of HOI4D.
    
    Args:
    h5path: path of h5 file to load.
    n_video: number of video to load from single h5 file.
    n_frame: number of frame to load from single video .
    save: save to file or not.
    dataset_name: name of folder to save npy file, only used if $save$ is True.
    
    Return:
    points_txyz: array of txyz coordinate with shape [n_video, t_video, n_point, 4]
    semantic: array of corresponding semantic label with shape [n_video, t_video, n_point]
    '''
    print(f"Loading {h5path}...")
    
    with h5py.File(h5path, 'r') as f:
        points_xyz = np.array(f['pcd'][: n_video, : n_frame])
        semantic = np.array(f['semantic'][: n_video, : n_frame])
        print('pcd', np.shape(points_xyz))
        print('semantic', np.shape(semantic))
    
    n_video, t_video, n_point, _ = points_xyz.shape
    print("Converting xyz to txyz")
    print(f'video num: {n_video}')
    print(f'frames per video: {t_video}')
    
    points_txyz = np.concatenate([np.zeros([n_video, t_video, n_point, 1]), points_xyz], axis=-1)
    points_txyz[:, :, :, 0] += (np.arange(t_video) + 1)[None, :, None]
    
    # points_txyz = points_txyz.reshape((n_video, t_video*n_point, 4))
    # semantic = semantic.reshape((n_video, t_video*n_point, 4))
    
    if save: 
        os.makedirs(dataset_name, exist_ok=True)
        np.save(f'{dataset_name}/pcd', points_txyz)
        np.save(f'{dataset_name}/semantic', semantic)
            
    return points_txyz, semantic


def load_bin(dir_path:str, config_path: str, dataset_name: str, mode: str, n_frame: int=-1, save: bool=False):
    '''
    Load binary file of KITTI.
    
    Args:
    dir_path: path to KITTI.
    config_path: path to KITTI config.
    dataset_name: name of folder to save npy file, only used if $save$ is True.
    mode: 'train' use 00-10; 'test' use 11-21.
    n_frame: number of frame to load from single video.
    save: save to file or not.
    
    
    Return:
    points_txyz: list of txyz coordinate with shape n_video * [t_video, n_point=130000, 4]
    depad_num: list of depad num, number of points in corresponding frames.
    semantic: list of corresponding semantic label with shape n_video * [t_video, n_point=13000]
    '''
    print(f"Loading {dir_path}...")
    
    path = dir_path + '/dataset/sequences/'
    videos = os.listdir(path)
    videos.sort()
    if mode == 'train': videos = videos[: 11]
    else: videos = videos[11:]
    print('Train, using data sequences', videos)
    
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
    
    if save: 
        print(f'saving to {dataset_name}')
        os.makedirs(dataset_name, exist_ok=True)
        np.save(f'{dataset_name}/pcd', points_txyz)
        np.save(f'{dataset_name}/semantic', semantic)
    
    return points_txyz, semantic, depad_num
    

def load_npy(dataset_name: str):
    '''
    Load npy file saved in func load_h5 or load_bin.
    
    Args:
    dataset_name: name of folder that saved npy files.
    NOTE: corresponding to $dataset_name$ in func load_h5 and load_bin.
    
    Return:
    points_txyz: array/list of txyz coordinate with shape [n_video, t_video, n_point, 3]
    semantic: array/list of corresponding semantic label with shape [n_video, t_video, n_point]
    '''
    print(f"Loading {dataset_name}...")
    
    points_txyz = np.load(f'{dataset_name}/pcd.npy')
    semantic = np.load(f'{dataset_name}/semantic.npy')
    
    return points_txyz, semantic
    
    
def construct_dataset(points_txyz, semantic, dataset_name: str, clip_length: int, depad_num = None):
    '''
    Construct dataset and config files.
    
    Args:
    points_txyz: array/list of txyz coordinate with shape [n_video, t_video, n_point, 4]
    semantic: array/list of corresponding semantic label with shape [n_video, t_video, n_point]
    dataset_name: name of folder to save clipped file, $batch size$ will be number of clipped file to read in a batch.
    clip_length: number of frame in a single clip.
    depad_num: if is not None, only take first depad[i] points in frame i.
    '''  
    print(f"Constructing dataset into {dataset_name}...")
    assert len(points_txyz) == len(semantic), 'semantic should be corresponding to points_txyz'
    
    os.makedirs(dataset_name, exist_ok=True)
    
    train_list = ''
    val_list = ''

    for v in range(len(points_txyz)): # loop over videos
        t_video, _, _ = points_txyz[v].shape
        N = t_video // clip_length
        
        if depad_num is None:
            pcd = points_txyz[v][: N*clip_length].reshape((N, -1, 4)) 
            label = semantic[v][: N*clip_length].reshape((N, -1))
        else: # depad
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
            np.savez(f'{dataset_name}/data_{i}.npz', points=pcd[i], labels=label[i])
            if i % 5 == 0:
                val_list += f'{dataset_name}/data_{i}.npz\n'
            else:
                train_list += f'{dataset_name}/data_{i}.npz\n'
        
        # left frames
        if t_video % clip_length > 0:
            if depad_num is None:
                pcd = points_txyz[v][N*clip_length: ].reshape((-1, 4))  
                label = semantic[v][N*clip_length: ].reshape((-1))  
            else:
                index = np.cumsum(depad_num[v][N*clip_length:])
                index = np.pad(index, (1,0), 'constant', constant_values=0)
                pcd = np.zeros((index[-1], 4))
                label = np.zeros((index[-1]))
                for k in range(t_video % clip_length):
                    id = N*clip_length + k
                    pcd[index[k]:index[k+1], :] = points_txyz[v][id][:depad_num[v][id]]
                    label[index[k]:index[k+1]] = semantic[v][id][:depad_num[v][id]]  
            np.savez(f'{dataset_name}/data_{N}.npz', points=pcd, labels=label)
            train_list += f'{dataset_name}/data_{N}.npz\n'
    # save config file
    f_train = open(f'{dataset_name}/train_data.txt', 'w')
    f_val = open(f'{dataset_name}/val_data.txt', 'w')
    f_train.write(train_list)
    f_train.close()
    f_val.write(val_list)
    f_val.close()



if __name__ == '__main__':
    
    def hoi4d():
        # example of building dataset using HOI4D
        clip_length = 8
        h5_dir = '/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5'
        dataset_name = f'/mnt/sdc/wangx/HexFormer/dataset/hoi4d/frame{clip_length}'
        # h5float64to32(h5_dir, 100) NOTE run only if you need to save memory
        points_txyz, semantic = load_h5(h5_dir+'/train1_float32.h5', dataset_name, 500, 160, True)
        construct_dataset(points_txyz, semantic, dataset_name, clip_length)
    
    def kitti():
        # TODO train, val, test split
        # example of building dataset using KITTI
        clip_length = 4
        kitti_dir = '/mnt/sdc/wangrh/data/SemanticKITTI'
        dataset_name = f'/mnt/sdc/wangx/HexFormer/dataset/kitti/frame{clip_length}'
        config_path = '/mnt/sdc/wangx/HexFormer/data_utils/config/semantic-kitti.yaml'
        points_txyz, semantic, depad_num = load_bin(kitti_dir, config_path, dataset_name, 'train', 100, True)
        construct_dataset(points_txyz, semantic, dataset_name, clip_length, depad_num)
        # load_bin(kitti_dir, dataset_name, 'train', -1, True)
        # load_bin(kitti_dir, dataset_name, 'test', -1, True)
    
    # hoi4d()
    kitti()

    