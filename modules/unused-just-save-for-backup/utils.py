# utils for data preprocess

import yaml
import torch
import numpy as np
from plyfile import PlyData

from hextree import Points, Hextree


def remap(semantic: np.array, cfg, inverse: bool = False):
    '''
    Remap semantic classes.
    
    Args:
    semantic: semantic classes to remap.
    cfg: KITTI config data.
    inverse: class2num if True, num2class if False. NOTE: See KITTI config for more.
    '''
    # DATA = yaml.safe_load(open(config_path, 'r'))

    # get number of interest classes, and the label mappings
    if inverse:
        print("Mapping xentropy to original labels")
        remapdict = cfg["learning_map_inv"]
    else:
        remapdict = cfg["learning_map"]

    # make lookup table for mapping
    maxkey = max(remapdict.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    return remap_lut[semantic]
    

class ReadPly:

    def __init__(self, has_normal: bool = True, has_color: bool = False,
                 has_label: bool = False):
        self.has_normal = has_normal
        self.has_color = has_color
        self.has_label = has_label

    def __call__(self, filename: str):
        plydata = PlyData.read(filename)
        vtx = plydata['vertex']

        output = dict()
        points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
        output['points'] = points.astype(np.float32)
        if self.has_normal:
            normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
            output['normals'] = normal.astype(np.float32)
        if self.has_color:
            color = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=1)
            output['colors'] = color.astype(np.float32)
        if self.has_label:
            label = vtx['label']
            output['labels'] = label.astype(np.int32)
        return output


class ReadNpz:

    def __init__(self, has_normal: bool = True, has_color: bool = False,
                 has_label: bool = False):
        self.has_normal = has_normal
        self.has_color = has_color
        self.has_label = has_label

    def __call__(self, filename: str):
        raw = np.load(filename)

        output = dict()
        output['points'] = raw['points'].astype(np.float32)
        if self.has_normal:
            output['normals'] = raw['normals'].astype(np.float32)
        if self.has_color:
            output['colors'] = raw['colors'].astype(np.float32)
        if self.has_label:
            output['labels'] = raw['labels'].astype(np.int32)
        return output


class ReadBin:
    # NOTE: noly used to load SemanticKITTI
    def __init__(self, has_label: bool = False):
        self.has_label = has_label
        self.config_path = '/mnt/sdc/wangx/HexFormer/data_utils/config/semantic-kitti-all.yaml'
        self.cfg = yaml.safe_load(open(self.config_path, 'r'))
        self.poses = []
        for i in range(22):
            filename = '/mnt/sdc/wangrh/data/SemanticKITTI/dataset/sequences/{:0>2d}/poses.txt'.format(i)
            pose = np.loadtxt(filename).reshape(-1, 3, 4).transpose((0, 2, 1))
            self.poses.append(pose)

    def __call__(self, filename: str):
        output = dict()
        root_pos = filename.find('/velodyne')
        assert root_pos > 0 # not found will be -1
        root_dir = filename[: root_pos]
        sequence_id = int(root_dir[-2:])
        frame_num = int(filename[root_pos+10: filename.find('.bin')])
        
        # point clouds
        pcds = []
        past_frame = max(frame_num - 3, 0)
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
            R, T = self.poses[sequence_id][i, :3], self.poses[sequence_id][i, -1]
            points[:, 1:4] = scan[:, 0:3] @ R + T    # get xyz
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
            output['labels'] = remap(sem_label, self.cfg)
        
        return output
    

class ReadFile:

    def __init__(self, has_normal: bool = False, has_color: bool = False,
                 has_label: bool = False):
        self.read_npz = ReadNpz(has_normal, has_color, has_label)
        self.read_ply = ReadPly(has_normal, has_color, has_label)
        self.read_bin = ReadBin(has_label)

    def __call__(self, filename: str):
        func = {'npz': self.read_npz, 'ply': self.read_ply, 'bin': self.read_bin}
        suffix = filename.split('.')[-1]
        return func[suffix](filename)


class Transform:
    ''' A boilerplate class which transforms an input data.
    The input data is first converted to :class:`Points`, then randomly transformed 
    (if enabled), and converted to an :class:`Hextree`.

    Args:
      depth (int): The hextree depth.
      full_depth (int): The hextree layers with a depth small than
          :attr:`full_depth` are forced to be full.
      distort (bool): If true, performs the data augmentation.
      angle (list): A list of 3 float values to generate random rotation angles.
      interval (list): A list of 3 float values to represent the interval of 
          rotation angles.
      scale (float): The maximum relative scale factor.
      uniform (bool): If true, performs uniform scaling.
      jittor (float): The maximum jitter values.
      orient_normal (str): Orient point normals along the specified axis, which is
          useful when normals are not oriented.
    '''

    # def __init__(self, depth: int, full_depth: int, distort: bool, angle: list,
    #              interval: list, scale: float, uniform: bool, jitter: float,
    #              flip: list, orient_normal: str = '', **kwargs):
    def __init__(self, depth: int, full_depth: int, distort: bool, angle: list,
                 interval: list, scale: float, flip: list, uniform: bool, **kwargs):
        super().__init__()

        # for hextree building
        self.depth = depth
        self.full_depth = full_depth

        # for data augmentation
        self.distort = distort
        self.angle = angle
        self.interval = interval
        self.scale = scale
        self.uniform = uniform
        # self.jitter = jitter
        self.flip = flip

        # for other transformations
        # self.orient_normal = orient_normal

    def __call__(self, sample: dict, idx: int):
        ''''''

        points = self.preprocess(sample, idx)
        output = self.transform(points, idx)
        output['hextree'] = self.points2hextree(output['points'])
        return output

    def preprocess(self, sample: dict, idx: int):
        ''' 
        Transforms :attr:`sample` to :class:`Points` and performs some specific
        transformations, like normalization.
        '''
        
        txyz = torch.from_numpy(sample['points'])
        points = Points(txyz)
        return points

    def transform(self, points: Points, idx: int):
        ''' 
        Applies the general transformations.
        '''

        # The augmentations including rotation, scaling.
        if self.distort:
            rng_angle, rng_scale, rnd_flip = self.rnd_parameters()
            
            points.rotate(rng_angle)
            points.scale_xyz(rng_scale)
            points.flip(rnd_flip)

        # if self.orient_normal:
        #     points.orient_normal(self.orient_normal)

        # !!! NOTE: Clip the point cloud to [-1, 1] before building the hextree
        # inbox_mask = points.clip_xyz(bbmin=-1, bbmax=1)
        # return {'points': points, 'inbox_mask': inbox_mask}
    
        return {'points': points}

    def points2hextree(self, points: Points):
        ''' 
        Converts the input :attr:`points` to an hextree.
        '''

        hextree = Hextree(self.depth, self.full_depth)
        hextree.build_hextree(points)
        return hextree

    def rnd_parameters(self):
        ''' 
        Generates random parameters for data augmentation.
        '''

        rnd_angle = [None] * 3
        for i in range(3):
            rot_num = self.angle[i] // self.interval[i]
            rnd = torch.randint(low=-rot_num, high=rot_num+1, size=(1,))
            rnd_angle[i] = rnd * self.interval[i] * (torch.pi / 180.0)
        rnd_angle = torch.cat(rnd_angle)

        rnd_scale = torch.rand(3) * (2 * self.scale) - self.scale + 1.0
        if self.uniform:
            rnd_scale[1] = rnd_scale[0]
            rnd_scale[2] = rnd_scale[0]

        rnd_flip = ''
        for i, c in enumerate('xyz'):
            if torch.rand([1]) < self.flip[i]:
                rnd_flip = rnd_flip + c

        return rnd_angle, rnd_scale, rnd_flip
