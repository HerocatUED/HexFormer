# build KITTI dataset for Solver

import yaml
import torch
import numpy as np

from hextree import Points
from thsolver import Dataset
from .utils import Transform


config_path = "configs/kitti_SemSeg/semantic-kitti-all.yaml"
cfg = yaml.safe_load(open(config_path, "r"))


def remap(semantic: np.array, inverse: bool = False):
        r"""
        Remap semantic classes.

        Args:
        semantic: semantic classes to remap.
        inverse: class2num if True, num2class if False. NOTE: See KITTI config for more.
        """
        
        # get number of interest classes, and the label mappings
        if inverse:
            # print("Mapping xentropy to original labels")
            remapdict = cfg["learning_map_inv"]
        else:
            remapdict = cfg["learning_map"]

        # make lookup table for mapping
        maxkey = max(remapdict.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        return remap_lut[semantic]


class KITTITransform(Transform):

    def __init__(self, flags):
        super().__init__(**flags)

        self.flags = flags
        self.scale_factor = 100

    def __call__(self, sample, idx=None):
        # get input sample
        pcds = Points(
            points=torch.from_numpy(sample["points"][:, :4]),
            labels=torch.from_numpy(sample["labels"]) if "labels" in sample.keys() else None,
            features=torch.from_numpy(sample["points"][:, 4:]),
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
        # pcds = self.align_z(pcds)

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
    


class ReadKITTI:
    def __init__(self, root_dir: str, has_label: bool = False, history: int = 3):
        self.has_label = has_label
        self.history = history
        self.poses = []
        for i in range(22):
            filename = root_dir + "/dataset/sequences/{:0>2d}/poses.txt".format(i)
            pose = np.loadtxt(filename).reshape(-1, 3, 4)
            self.poses.append(pose)
        self.cam2vel = np.array(
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0.08], [0, 0, 0, 1]]
        )
        self.vel2cam = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.08], [0, 0, 0, 1]]
        )

    def __call__(self, filename: str):
        output = dict()
        root_pos = filename.find("/velodyne")
        assert root_pos > 0  # not found will be -1
        root_dir = filename[:root_pos]
        sequence_id = int(root_dir[-2:])
        frame_num = int(filename[root_pos + 10 : filename.find(".bin")])
        past_frame = max(frame_num - self.history, 0)
        
        # point clouds
        pcds = []
        for j, i in enumerate(range(past_frame, frame_num + 1)):
            scan_name = root_dir + "/velodyne/{:0>6d}.bin".format(i)
            scan = np.fromfile(scan_name, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            N = np.shape(scan)[0]
            points = np.ones((N, 8), dtype=np.float32)
            # put in attribute
            points[:, 1:4] = self.local2global(scan[:, 0:3], i, sequence_id)  # get xyz
            points[:, 4] = scan[:, 3]  # density
            points[:, 5:] = self.polar(scan[:, 0:3]) # polar
            points[:, 0] *= j
            pcds.append(points)
        output["points"] = np.vstack(pcds)

        # label
        if self.has_label:
            labels = []
            for i in range(past_frame, frame_num + 1):
                label_name = root_dir + "/labels/{:0>6d}.label".format(i)
                label = np.fromfile(label_name, dtype=np.int32)
                label = label.reshape((-1))
                sem_label = label & 0xFFFF  # semantic label in lower half
                inst_label = label >> 16  # instance id in upper half
                # sanity check
                assert (sem_label + (inst_label << 16) == label).all()
                labels.append(sem_label)
            output["labels"] = np.hstack(labels)
            output["labels"] = remap(output["labels"], False)

        return output

    def local2global(self, pcd: np.array, frame_id: int, sequence_id: int):
        r"""
        Trans local coordinates to global coordinates.

        Args:
        pcd: local xyz coordinates.
        frame_id: ID of frame that point cloud belones to.
        sequence_id: ID of sequence that point cloud belones to.
        """
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
    
    def polar(self, pcd: np.array):
        r"""
        Trans local coordinates to polar coordinates.

        Args:
        pcd: local xyz coordinates.
        """
        x = pcd[:, 0]
        y = pcd[:, 1]
        z = pcd[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / (r + 1e-10))
        polar_coords = np.stack((r, theta, phi), axis = -1)
        return polar_coords


class CollateBatch:

    def __init__(self, cutmix: int = 0.5):
        super().__init__()
        self.cutmix = cutmix

    def __call__(self, batch: list):
        assert type(batch) == list

        # a list of dicts -> a dict of lists
        outputs = {key: [b[key] for b in batch] for key in batch[0].keys()}

        return outputs


def get_kitti_sem_seg_dataset(flags):
    transform = KITTITransform(flags)
    read_file = ReadKITTI(
        has_label=flags.has_label, root_dir=flags.location, history=flags.history
    )
    collate_batch = CollateBatch(flags.cutmix)

    dataset = Dataset(flags.location, flags.filelist, transform, read_file=read_file)
    return dataset, collate_batch
