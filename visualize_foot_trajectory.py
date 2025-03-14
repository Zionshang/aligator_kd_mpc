import numpy as np

import matplotlib.pyplot as plt

def load_points(file_path):
    # 假设每行数据为 x,y,z 格式，无标题
    return np.loadtxt(file_path, delimiter=',')

def main():
    # 读取数据
    logger_points = load_points("/home/zishang/cpp_workspace/aligator_kd_mpc/build/lf_foot_logger.csv")
    ref_logger_points = load_points("/home/zishang/cpp_workspace/aligator_kd_mpc/build/lf_foot_ref_logger.csv")
    
    # 设置三维绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制数据点
    ax.scatter(logger_points[:, 0], logger_points[:, 1], logger_points[:, 2],
               c='red', marker='o', label='Logger')
    ax.scatter(ref_logger_points[:, 0], ref_logger_points[:, 1], ref_logger_points[:, 2],
               c='blue', marker='^', label='Ref Logger')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Foot Trajectory')
    ax.legend()
    
    plt.show()

if __name__ == '__main__':
    main()