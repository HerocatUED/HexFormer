# Visualize the result of semantic segmantation

import numpy as np
import matplotlib.pyplot as plt

def visualize_point_cloud(points, mode, class_num:int = 26):
    # 49 for hoi4d, 26 for kitti
    colors = plt.cm.jet(np.linspace(0, 1, class_num))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(class_num):
        mask = points[:, 3] == i
        if np.any(mask):
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], c=colors[i].reshape(1,-1), label=f"Category {i}", s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(mode)
    ax.legend(loc="upper right", ncol=4)
    
def print_range(pcds):
    x = pcds[:][:][0]
    y = pcds[:][:][1]
    z = pcds[:][:][2]
    print('x:', np.max(x), np.min(x))
    print('y:', np.max(y), np.min(y))
    print('z:', np.max(z), np.min(z))
    
    
def trans_visualize(rand_ids, logdir):
    '''
    script for visualization
    '''
    for i, rand_id in enumerate(rand_ids):
        points = np.array(np.load(f"logs/log_{logdir}_kitti/result_sample/points_{rand_id}.npz")['arr_0'])
        mask = points[:, 0] == 0
        points = points[mask]
        points = points[:, 1:]
        print("points", np.shape(points))
        pts_num = np.shape(points)[0]
        pred = np.array(np.load(f"logs/log_{logdir}_kitti/result_sample/pred_{rand_id}.npz")['arr_0'])
        pred = pred[:pts_num]
        print("pred", np.shape(pred))
        label = np.array(np.load(f"logs/log_{logdir}_kitti/result_sample/label_{rand_id}.npz")['arr_0'])
        label = label[:pts_num]
        print("label", np.shape(label))
        prediction = np.concatenate([points, np.expand_dims(pred, 1)], axis=-1)
        groundtruth = np.concatenate([points, np.expand_dims(label, 1)], axis=-1)
        np.save(f"dataset/visualize/prediction_{i}.npy", prediction)
        np.save(f"dataset/visualize/groundtruth_{i}.npy", groundtruth)  
        

if __name__ == '__main__':
    # TODO: add annotations to functions
    trans_visualize(["135.0000"], "all")
    pred = np.load('dataset/visualize/prediction_0.npy')
    # print_range(pcds)
    visualize_point_cloud(pred, 'predict')
    
    gt = np.load('dataset/visualize/groundtruth_0.npy')
    # print_range(pcds_gt)
    visualize_point_cloud(gt, 'groundtruth')
    
    plt.draw()
    plt.show()
    # plt.savefig('out1_0.png')
    # plt.pause(10)
    # plt.close() 