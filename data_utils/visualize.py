# Visualize the result of semantic segmantation
import yaml
import numpy as np
import matplotlib.pyplot as plt

    
def print_range(pcds):
    x = pcds[:][:][0]
    y = pcds[:][:][1]
    z = pcds[:][:][2]
    print('x:', np.max(x), np.min(x))
    print('y:', np.max(y), np.min(y))
    print('z:', np.max(z), np.min(z))
    
    
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
     
        
def kitti_vis(points: np.array, config_path: str, mode: str):
    '''
    Args:
    points: point clouds (x, y, z, class)
    config_path: path to KITTI config.
    mode: pred or gt
    '''
    DATA = yaml.safe_load(open(config_path, 'r'))

    remapdict = DATA["learning_map_inv"]
    color_map = DATA["color_map"]

    # make lookup table for mapping
    maxkey = max(remapdict.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(49):
        mask = points[:, 3] == i
        inv_class = remap_lut[i]
        color = color_map[inv_class]
        if np.any(mask):
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], c=color.reshape(1,-1), label=f"Category {i}", s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(mode)
    ax.legend(loc="upper right", ncol=4)
        

if __name__ == '__main__':
    # TODO: add annotations to functions
    
    trans(["120.0000"], "all", "kitti")
    
    # pred = np.load('dataset/visualize/prediction_0.npy')
    # # print_range(pcds)
    # visualize_point_cloud(pred, 'predict')
    
    # gt = np.load('dataset/visualize/groundtruth_0.npy')
    # # print_range(pcds_gt)
    # visualize_point_cloud(gt, 'groundtruth')
    
    # plt.draw()
    # plt.show()
