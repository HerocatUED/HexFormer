import numpy as np
import torch
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import h5py

index_to_label = np.arange(0, 49, dtype='int32')
label_to_index = np.arange(0, 49, dtype='int32')
index_to_class = [str(i) for i in range(0, 49)]


def index_to_label_func(x):
    return index_to_label[x]


index_to_label_vec_func = np.vectorize(index_to_label_func)


class SegDataset(Dataset):
    def __init__(self, root: str, frames_per_clip=3, num_points=8192, train=True):
        super(SegDataset, self).__init__()

        self.frames_per_clip = frames_per_clip
        self.train = train
        self.num_points = num_points

        # for a single .h5 file:
        # pcd       (750, 300, 8192, 3)
        # center    (750, 300, 3)
        # semantic  (750, 300, 8192)

        self.data_file = []
        if self.train:
            for filename in ['train1.h5', 'train2.h5', 'train3.h5', 'train4.h5']:
                self.data_file.append(h5py.File(root + '/' + filename, 'r'))
                print(f'{filename}')
        else:
            for filename in ['test.h5']:
                self.data_file.append(h5py.File(root + '/' + filename, 'r'))
                print(f'{filename}')

    def __len__(self):
        leng = 0
        if self.train:
            leng = 2971 * 100  # 750 + 750 + 750 + 721
        else:
            leng = 500 * 100
        return leng

    def read_training_data_point(self, index):
        frame_idx = index % 100
        frame_id = frame_idx * 3

        if self.train:
            idx = int(index / 100)
            s = int(idx / 750)
            d = idx % 750
        else:
            idx = int(index / 100)
            s = int(idx / 500)
            d = idx % 500

        data = self.data_file[s]
        pcd = data['pcd'][d][frame_id:int(frame_id + self.frames_per_clip)]
        rgb = data['pcd'][d][frame_id:int(frame_id + self.frames_per_clip)]
        semantic = data['semantic'][d][frame_id:int(
            frame_id + self.frames_per_clip)]
        center_0 = data['center'][d][frame_id]

        return pcd, rgb, semantic, center_0

    def augment(self, pcd, center):
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pcd = pcd - center
            jittered_data = np.clip(
                0.01 * np.random.randn(self.frames_per_clip, self.num_points, 3), -1 * 0.05, 0.05)
            jittered_data += pcd
            pcd = pcd + center

        scale = np.random.uniform(0.8, 1.2)
        pcd = (pcd - center) * scale + center

        rot_axis = np.array([0, 1, 0])
        rot_angle = np.random.uniform(np.pi * -0.05, np.pi * 0.05)
        q = Quaternion(axis=rot_axis, angle=rot_angle)
        R = q.rotation_matrix

        pcd = np.dot(pcd - center, R) + center
        return pcd

    def label_conversion(self, semantic):
        labels = []
        for i, s in enumerate(semantic):
            sem = s.astype('int32')
            label = index_to_label_vec_func(sem)

            labels.append(label)
        return labels

    def choice_to_num_points(self, pc, rgb, label):
        # shuffle idx to change point order (change FPS behavior)
        for f in range(self.frames_per_clip):
            idx = np.arange(pc[f].shape[0])
            choice_num = self.num_points
            if pc[f].shape[0] > choice_num:
                shuffle_idx = np.random.choice(idx, choice_num, replace=False)
            else:
                shuffle_idx = np.concatenate(
                    [np.random.choice(idx, choice_num - idx.shape[0]), np.arange(idx.shape[0])])
            pc[f] = pc[f][shuffle_idx]
            rgb[f] = rgb[f][shuffle_idx]
            label[f] = label[f][shuffle_idx]

        pc = np.stack(pc, axis=0)
        rgb = np.stack(rgb, axis=0)
        label = np.stack(label, axis=0)

        return pc, rgb, label

    def __getitem__(self, index):
        pc, rgb, semantic, center = self.read_training_data_point(index)

        label = self.label_conversion(semantic)
        label = np.array(label)
        if self.train:
            pcd = self.augment(pcd, center)
        rgb = np.swapaxes(rgb, 1, 2)

        return pc.astype(np.float32), rgb.astype(np.float32), label.astype(np.int64)


if __name__ == '__main__':

    print("Creating data loaders")

    datasets = SegDataset(
        root='/mnt/sdc/wangx/HOI4D_dataset/seg_data_h5', frames_per_clip=3, train=True)

    data_loader = torch.utils.data.DataLoader(
        datasets, batch_size=24, shuffle=True, num_workers=8, pin_memory=False)
