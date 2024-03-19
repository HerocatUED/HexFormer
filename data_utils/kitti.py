# build KITTI dataset for Solver

import yaml
import torch
import numpy as np

from hextree import Points
from thsolver import Dataset
from .utils import Transform


# def align_z(points: Points):
#     points.points[:, 3] -= points.points[:, 3].min()
#     return points


def rand_crop(points: Points, max_npt: int):
    r''' Keeps `max_npt` pts at most centered by a radomly chosen pts. 
    '''

    pts = points.points
    npt = points.npt
    crop_mask = torch.ones(npt, dtype=torch.bool)
    if npt > max_npt:
        rand_idx = torch.randint(low=0, high=npt, size=(1,))
        sort_idx = torch.argsort(torch.sum(((pts - pts[rand_idx])[1:])**2, 1))
        crop_idx = sort_idx[max_npt:]
        crop_mask[crop_idx] = False
        points = points[crop_mask]
    return points, crop_mask


class KITTITransform(Transform):

    def __init__(self, flags):
        super().__init__(**flags)

        self.flags = flags

    def __call__(self, sample, idx=None):
        # construct and normalize points
        pcds = Points(points=torch.from_numpy(sample['points'][:, :4]),
                      labels=torch.from_numpy(sample['labels']),
                      features=torch.from_numpy(sample['points'][:, 4:]))
        pcds.normalize_xyz(keep_shape=True)

        # transform including rotatation, translation, scaling, and flipping
        output = self.transform(pcds, idx)   # points and inbox_mask
        points = output['points']

        # align z
        # points = align_z(points)
        return {'points': points}


class ReadKITTI:
    def __init__(self, kitti_dir: str, has_label: bool = False, history: int = 3):
        self.has_label = has_label
        self.history = history
        self.config_path = 'data_utils/config/semantic-kitti-all.yaml'
        self.cfg = yaml.safe_load(open(self.config_path, 'r'))
        self.poses = []
        for i in range(22):
            filename = kitti_dir + '/dataset/sequences/{:0>2d}/poses.txt'.format(i)
            pose = np.loadtxt(filename).reshape(-1, 3, 4)
            self.poses.append(pose)
        self.cam2vel = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0.08],
            [0, 0, 0, 1]
        ])
        self.vel2cam = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, -0.08],
            [0, 0, 0, 1]
        ])

    def __call__(self, filename: str):
        output = dict()
        root_pos = filename.find('/velodyne')
        assert root_pos > 0 # not found will be -1
        root_dir = filename[: root_pos]
        sequence_id = int(root_dir[-2:])
        frame_num = int(filename[root_pos+10: filename.find('.bin')])
        
        # point clouds
        pcds = []
        past_frame = max(frame_num - self.history, 0)
        if frame_num == 2: 
            past_frame = 1
        j = 0
        for i in range(past_frame, frame_num + 1):
            scan_name = root_dir + '/velodyne/{:0>6d}.bin'.format(i)
            scan = np.fromfile(scan_name, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            N = np.shape(scan)[0]
            points = np.ones((N, 5), dtype=np.float32)
            # put in attribute
            points[:, 1:4] = self.local2global(scan[:, 0:3], i, sequence_id)    # get xyz
            points[:, 4] = scan[:, 3] # density
            points[:, 0] *= j
            pcds.append(points) 
            j = j + 1
        output['points'] = np.vstack(pcds)
        
        # label
        if self.has_label:
            label_name = root_dir + '/labels/{:0>6d}.label'.format(frame_num)
            label = np.fromfile(label_name, dtype=np.uint32)
            label = label.reshape((-1))
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16    # instance id in upper half
            # sanity check
            assert((sem_label + (inst_label << 16) == label).all())
            output['labels'] = self.remap(sem_label)
        
        return output
    
    def local2global(self, pcd: np.array, frame_id: int, sequence_id: int):
        '''
        Trans local coordinates to global coordinates.
        
        Args:
        pcd: local xyz coordinates.
        frame_id: ID of frame that point cloud belones to.
        sequence_id: ID of sequence that point cloud belones to.
        '''
        # cam2cam transition matrix
        matrix = np.zeros((4, 4))
        matrix[:3] = self.poses[sequence_id][frame_id]
        matrix[3, 3] = 1
        # prepare local_xyz matrix
        local_xyz = np.ones((np.shape(pcd)[0], 4))
        local_xyz[:, :3] = pcd
        local_xyz = np.expand_dims(local_xyz, axis=-1)
        # vel2cel transition matrix
        trans_matrix = self.cam2vel @ matrix @ self.vel2cam
        # global_xyz
        global_xyz = (trans_matrix @ local_xyz).reshape((-1, 4))
        
        return global_xyz[:, :3]
    
    def remap(self, semantic: np.array, inverse: bool = False):
        '''
        Remap semantic classes.
        
        Args:
        semantic: semantic classes to remap.
        inverse: class2num if True, num2class if False. NOTE: See KITTI config for more.
        '''
        
        # get number of interest classes, and the label mappings
        if inverse:
            print("Mapping xentropy to original labels")
            remapdict = self.cfg["learning_map_inv"]
        else:
            remapdict = self.cfg["learning_map"]

        # make lookup table for mapping
        maxkey = max(remapdict.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        return remap_lut[semantic]


class CollateBatch:

    def __init__(self, cutmix: int = 0.5):
        super().__init__()
        self.cutmix = cutmix

    def __call__(self, batch: list):
        assert type(batch) == list

        # a list of dicts -> a dict of lists
        outputs = {key: [b[key] for b in batch] for key in batch[0].keys()}

        return outputs


def get_kitti_seg_dataset(flags):
    transform = KITTITransform(flags)
    read_file = ReadKITTI(has_label=flags.has_label, kitti_dir=flags.location, history=flags.history)
    collate_batch = CollateBatch(flags.cutmix)

    dataset = Dataset(flags.location, flags.filelist,
                      transform, read_file=read_file)
    return dataset, collate_batch


def construct_filelist(kitti_dir:str, dataset_dir: str):
    '''
    Construct filelist.
    mode: 'train' use 00-10; 'test' use 11-21.
    
    Args:
    kitti_dir: path to KITTI.
    dataset_dir: path to save filelist.
    '''  
    import os
    
    train_list = ''
    val_list = ''
    test_list = ''

    path = kitti_dir + '/dataset/sequences/'
    videos = os.listdir(path)
    videos.sort()
    for video in videos:
        pcd_dir = path + video + '/velodyne/'
        pcd_files = os.listdir(pcd_dir)
        pcd_files.sort()
        if int(video) >= 11:
            for pcd in pcd_files:
                test_list += pcd_dir + pcd + '\n'
        elif int(video) == 8:
            for pcd in pcd_files:
                val_list += pcd_dir + pcd + '\n'
        else:
            for pcd in pcd_files:
                train_list += pcd_dir + pcd + '\n'
                
    f_train = open(f'{dataset_dir}/train_data.txt', 'w')
    f_train.write(train_list)
    f_train.close()
    f_val = open(f'{dataset_dir}/val_data.txt', 'w')
    f_val.write(val_list)
    f_val.close()
    f_test = open(f'{dataset_dir}/test_data.txt', 'w')
    f_test.write(test_list)
    f_test.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=False, default='dataset/kitti')
    args = parser.parse_args()
    
    construct_filelist(args.kitti_dir, args.save_dir)