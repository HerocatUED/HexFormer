# build HOI4D dataset for Solver

import torch
import numpy as np
from thsolver import Dataset

from hextree import Points
from .utils import Transform

def remap(semantic: np.array, inverse: bool = False):
    r"""
        Remap semantic classes.

        Args:
        semantic: semantic classes to remap.
        inverse: class2num if True, num2class if False. NOTE: See KITTI config for more.
        """
    return semantic

    
class HOI4DTransform(Transform):

    def __init__(self, flags):
        super().__init__(**flags)

        self.flags = flags
        self.scale_factor = 100

    def __call__(self, sample, idx=None):
        # get input sample
        pcds = Points(
            points=torch.from_numpy(sample["points"][:, :4]),
            labels=torch.from_numpy(sample["labels"]) if "labels" in sample.keys() else None,
        )
        
        # normalize points
        pcds.normalize_xyz(keep_shape=True, box_size=2*self.scale_factor)
        
        # transform including rotatation, translation, scaling, and flipping
        pcds = self.transform(pcds, idx)
        
        # random crop
        if self.distort:
            max_npt = self.flags.max_npt if self.flags.max_npt > 0 else pcds.npt
            max_npt = min(max_npt, int(pcds.npt * self.flags.crop_ratio))
            pcds = self.rand_crop(pcds, max_npt)
            
        # align z
        pcds = self.align_z(pcds)

        return {"points": pcds}
    
    def rand_crop(self, points: Points, max_npt: int):
        r''' Keeps `max_npt` pts at most centered by a radomly chosen pts. 
        '''
        pts = points.points
        npt = points.npt
        crop_mask = torch.ones(npt, dtype=torch.bool)
        if npt > max_npt:
            rand_idx = torch.randint(low=0, high=npt, size=(1,))
            sort_idx = torch.argsort(torch.sum((pts - pts[rand_idx])**2, 1))
            crop_idx = sort_idx[max_npt:]
            crop_mask[crop_idx] = False
            points = points[crop_mask] 
        return points
    
    def align_z(self, points: Points):
        points.points[:, 3] -= points.points[:, 3].min()
        return points


class ReadHOI4D:
    def __init__(self, root_dir: str, has_label: bool = False, history: int = 3):
        self.has_label = has_label
        self.history = history

    def __call__(self, filename: str):
        output = dict()
        root_pos = filename.find("/velodyne")
        assert root_pos > 0  # not found will be -1
        root_dir = filename[:root_pos]
        frame_num = int(filename[root_pos + 10 : filename.find(".bin")])

        # point clouds
        pcds = []
        past_frame = max(frame_num - self.history, 0)
        for j, i in enumerate(range(past_frame, frame_num + 1)):
            scan_name = root_dir + "/velodyne/{:0>6d}.bin".format(i)
            scan = np.fromfile(scan_name, dtype=np.float32)
            scan = scan.reshape((-1, 3))
            N = np.shape(scan)[0]
            points = np.ones((N, 4), dtype=np.float32)
            # put in attribute
            points[:, 1:] = scan  # get xyz
            points[:, 0] *= j
            pcds.append(points)
        output["points"] = np.vstack(pcds)

        # label
        if self.has_label:
            label_name = root_dir + "/labels/{:0>6d}.label".format(frame_num)
            label = np.fromfile(label_name, dtype=np.int32)
            sem_label = label.reshape((-1))
            output["labels"] = remap(sem_label, False)

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


def get_hoi4d_act_seg_dataset(flags):
    transform = HOI4DTransform(flags)
    read_file = ReadHOI4D(
        has_label=flags.has_label, root_dir=flags.location, history=flags.history
    )
    collate_batch = CollateBatch(flags.cutmix)

    dataset = Dataset(flags.location, flags.filelist, transform, read_file=read_file)
    return dataset, collate_batch
