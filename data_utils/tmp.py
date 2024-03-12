# Visualize the result of semantic segmantation
import yaml
import numpy as np
import matplotlib.pyplot as plt

cam2vel = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0.08],
            [0, 0, 0, 1]
        ])
vel2cam = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, -0.08],
    [0, 0, 0, 1]
])

def local2global(pose, pcd: np.array, frame_id: int, sequence_id: int):
        '''
        Trans local coordinates to global coordinates.
        
        Args:
        pcd: local xyz coordinates.
        frame_id: ID of frame that point cloud belones to.
        sequence_id: ID of sequence that point cloud belones to.
        '''
        matrix = np.zeros((4, 4))
        matrix[:3] = pose[frame_id]
        matrix[3, 3] = 1
        local_xyz = np.ones((np.shape(pcd)[0], 4))
        local_xyz[:, :3] = pcd
        local_xyz = np.expand_dims(local_xyz, axis=-1)
        trans_matrix = cam2vel @ matrix @ vel2cam
        global_xyz = (trans_matrix @ local_xyz).reshape((-1, 4))
        return global_xyz[:, :3]

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
    
    
    
def trans(rand_ids, logdir, dataset:str="kitti"):
    '''
    trans t,x,y,z to x,y,z
    '''

    for i, rand_id in enumerate(rand_ids):
        points = np.array(np.load(f"logs/log_{logdir}_{dataset}/result_sample/points_{rand_id}.npz")['arr_0'])
        mask = points[:, 0] == 0
        points = points[mask]
        points = points[:, 1:]
        print("points", np.shape(points))
        pts_num = np.shape(points)[0]
        pred = np.array(np.load(f"logs/log_{logdir}_{dataset}/result_sample/pred_{rand_id}.npz")['arr_0'])
        pred = pred[:pts_num]
        print("pred", np.shape(pred))
        label = np.array(np.load(f"logs/log_{logdir}_{dataset}/result_sample/label_{rand_id}.npz")['arr_0'])
        label = label[:pts_num]
        print("label", np.shape(label))
        prediction = np.concatenate([points, np.expand_dims(pred, 1)], axis=-1)
        groundtruth = np.concatenate([points, np.expand_dims(label, 1)], axis=-1)
        np.save(f"dataset/visualize/prediction_{logdir}_{i}.npy", prediction)
        np.save(f"dataset/visualize/groundtruth_{logdir}_{i}.npy", groundtruth)  
       
       
def hoi4d_vis(points, mode):
    colors = plt.cm.jet(np.linspace(0, 1, 49))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(49):
        mask = points[:, 3] == i
        if np.any(mask):
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], c=colors[i].reshape(1,-1), label=f"Category {i}", s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(mode)
    ax.legend(loc="upper right", ncol=4) 
     
 
def kitti_pcd(filename: str):
    config_path = '/mnt/sdc/wangx/HexFormer/data_utils/config/semantic-kitti-all.yaml'
    cfg = yaml.safe_load(open(config_path, 'r'))
    posefile = '/mnt/sdc/wangrh/data/SemanticKITTI/dataset/sequences/{:0>2d}/poses.txt'.format(0)
    pose = np.loadtxt(posefile).reshape(-1, 3, 4)
    
    
    output = dict()
    root_pos = filename.find('/velodyne')
    assert root_pos > 0 # not found will be -1
    root_dir = filename[: root_pos]
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
        points = np.ones((N, 4), dtype=np.float32)
        # put in attribute
        points[:, 1:4] = local2global(pose, scan[:, 0:3], i, 0) # get xyz
        # points[:, 4:] = scan[:, 3:] # density
        points[:, 0] *= j
        pcds.append(points) 
        j = j + 1
    output['points'] = np.vstack(pcds)
    
    # label
    l = -1 * np.ones(np.shape(output['points'])[0])
    label_name = root_dir + '/labels/{:0>6d}.label'.format(frame_num)
    label = np.fromfile(label_name, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16    # instance id in upper half
    # sanity check
    assert((sem_label + (inst_label << 16) == label).all())
    output['labels'] = remap(sem_label, cfg)
    l[-np.shape(output['labels'])[0]:] = output['labels']
    
    np.savez('tmp.npz', p=output['points'], l=l)
    return output

if __name__ == '__main__':
    # TODO: add annotations to functions
    kitti_pcd('/mnt/sdc/wangrh/data/SemanticKITTI/dataset/sequences/00/velodyne/000008.bin')
    
    # trans(["120.0000"], "all", "kitti")
    
    # pred = np.load('dataset/visualize/prediction_0.npy')
    # # print_range(pcds)
    # visualize_point_cloud(pred, 'predict')
    
    # gt = np.load('dataset/visualize/groundtruth_0.npy')
    # # print_range(pcds_gt)
    # visualize_point_cloud(gt, 'groundtruth')
    
    # plt.draw()
    # plt.show()
