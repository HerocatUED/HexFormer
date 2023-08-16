import numpy as np
import random

# pcd = np.load('./output/predict/train1_train4_150.npy')
# print(np.shape(pcd))

# pcd = pcd.reshape(-1, 300, 8192, 4)
# for i in range(5):
#     index = random.randint(0,299)
#     pred = pcd[i*10][index][:][-1]
#     coord = pcd[i*10][index][:][:-1]
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(points):
    # 使用49个不同的颜色
    colors = plt.cm.jet(np.linspace(0, 1, 49))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(49):
        mask = points[:, 3] == i
        if np.any(mask):
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], c=colors[i].reshape(1,-1), label=f"Category {i}", s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc="upper right", ncol=2)
    plt.draw()  # 画图
    plt.show()
    # plt.pause(10)  # 显示10秒
    # plt.close()  # 关闭图形

if __name__ == '__main__':
    # 示例点云数据
    points = np.array([
        [1.2, 2.6, 0.7, 13],
        [1.5, 2.5, 0.6, 12],
        [2.2, 2.4, 1.7, 13],
        # ... (其他点)
    ])

    visualize_point_cloud(points)