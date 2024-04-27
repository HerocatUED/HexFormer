# build HOI4D dataset for Solver

import torch
import h5py
import numpy as np
from thsolver import Dataset

from hextree import Points
from .utils import Transform


class HOI4DTransform(Transform):

    def __init__(self, flags):
        super().__init__(**flags)

        self.flags = flags
        self.scale_factor = 5.12 # TODO

    def __call__(self, sample, idx=None):
        # get input sample
        pcds = Points(
            points=torch.from_numpy(sample["points"][:, :4]),
            labels=torch.from_numpy(sample["labels"]),
        )
        
        # normalize points
        pcds.normalize_xyz(keep_shape=True, box_size=2*self.scale_factor)
        
        # transform including rotatation, translation, scaling, and flipping
        output = self.transform(pcds, idx)

        return output


class ReadHOI4D:
    def __init__(self, root_dir: str, has_label: bool = False, history: int = 3):
        self.has_label = has_label
        self.history = history
        self.datas = []
        if self.has_label:
            for filename in ['train1_float32.h5', 'train2_float32.h5', 'train3_float32.h5', 'train4_float32.h5']:
                self.datas.append(h5py.File(root_dir + '/' + filename, 'r'))
        else:
            for filename in ['test.h5']:
                self.datas.append(h5py.File(root_dir + '/' + filename, 'r'))
        
        # for a single .h5 file (except the last one, only 721)
        # pcd      (750, 300, 8192, 3)
        # center   (750, 300, 3)
        # semantic (750, 300, 8192)
        
    def __call__(self, filename: str):
        
        # index conversion
        frame_cnt = int(filename.split('/')[-1]) # frame id in all data
        frame_id = int(frame_cnt % 300) # frame id in a single video
        past_frame = max(frame_id - self.history, 0)
        
        if self.has_label:
            idx = int(frame_cnt / 300) # video id in all data
            block_id = int(idx / 750) # block idx
            video_id = idx % 750 # video id in a single block
        else:
            idx = int(frame_cnt / 300)
            block_id = int(idx / 500)
            video_id = idx % 500
            
        # extract data
        data = self.datas[block_id]
        xyz = data['pcd'][video_id][past_frame: frame_id+1]
        label = data['semantic'][video_id][frame_id]

        # convert xyz to txyz
        points_txyz = np.concatenate([np.zeros([self.history + 1, 8192, 1]), xyz], axis=-1)
        points_txyz[:, :, 0] += np.arange(self.history + 1)[:, None]
        points_txyz = points_txyz.reshape((-1, 4))  # (t_video * n_point, 4)
        label = label.reshape((-1,))
    
        # construct output
        output = dict()
        output["points"] = points_txyz.astype(np.float32)
        output["labels"] = label.astype(np.int32)

        return output


class CollateBatch:

    def __init__(self, cutmix: int = 0.5):
        super().__init__()
        self.cutmix = cutmix

    def __call__(self, batch: list):
        assert type(batch) == list

        # a list of dicts -> a dict of lists
        outputs = {key: [b[key] for b in batch] for key in batch[0].keys()}

        return outputs


def get_hoi4d_seg_dataset(flags):
    transform = HOI4DTransform(flags)
    read_file = ReadHOI4D(
        has_label=flags.has_label, root_dir=flags.location, history=flags.history
    )
    collate_batch = CollateBatch(flags.cutmix)

    dataset = Dataset(flags.location, flags.filelist, transform, read_file=read_file)
    return dataset, collate_batch
