# Written by Xiang Wang

import numpy as np
import matplotlib.pyplot as plt

def visualize_point_cloud(points, mode):
    # 使用49个不同的颜色
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
    
def print_range(pcds):
    x = pcds[:][:][0]
    y = pcds[:][:][1]
    z = pcds[:][:][2]
    print('x:', np.max(x), np.min(x))
    print('y:', np.max(y), np.min(y))
    print('z:', np.max(z), np.min(z))

if __name__ == '__main__':
    # 示例点云数据
    pcds = np.load('train1_train1_sample.npy')
    points = pcds[0]
    # print_range(pcds)
    visualize_point_cloud(points, 'predict')
    
    pcds_gt = np.load('train1_sample_gt.npy')
    points_gt = pcds_gt[0]
    # print_range(pcds_gt)
    visualize_point_cloud(points_gt, 'groundtruth')
    
    plt.draw()  # 画图
    plt.show()
    # plt.savefig('out1_0.png')
    # plt.pause(10)  # 显示10秒
    # plt.close()  # 关闭图形