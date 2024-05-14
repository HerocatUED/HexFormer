# build KITTI dataset for Solver

import yaml
import torch
import numpy as np

from hextree import Points
from thsolver import Dataset
from .utils import Transform


config_path = "config/kitti/semantic-kitti-all.yaml"
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
            labels=torch.from_numpy(sample["labels"]),
            features=torch.from_numpy(sample["points"][:, 4:]),
        )
        
        # normalize points
        pcds.normalize_xyz(keep_shape=True, box_size=2*self.scale_factor)
        
        # transform including rotatation, translation, scaling, and flipping
        output = self.transform(pcds, idx)

        return output


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

        # point clouds
        pcds = []
        past_frame = max(frame_num - self.history, 0)
        for j, i in enumerate(range(past_frame, frame_num + 1)):
            scan_name = root_dir + "/velodyne/{:0>6d}.bin".format(i)
            scan = np.fromfile(scan_name, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            N = np.shape(scan)[0]
            points = np.ones((N, 5), dtype=np.float32)
            # put in attribute
            points[:, 1:4] = self.local2global(scan[:, 0:3], i, sequence_id)  # get xyz
            points[:, 4] = scan[:, 3]  # density
            points[:, 0] *= j
            pcds.append(points)
        output["points"] = np.vstack(pcds)

        # label
        if self.has_label:
            label_name = root_dir + "/labels/{:0>6d}.label".format(frame_num)
            label = np.fromfile(label_name, dtype=np.int32)
            label = label.reshape((-1))
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16  # instance id in upper half
            # sanity check
            assert (sem_label + (inst_label << 16) == label).all()
            output["labels"] = remap(sem_label, False)

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
    read_file = ReadKITTI(
        has_label=flags.has_label, root_dir=flags.location, history=flags.history
    )
    collate_batch = CollateBatch(flags.cutmix)

    dataset = Dataset(flags.location, flags.filelist, transform, read_file=read_file)
    return dataset, collate_batch
