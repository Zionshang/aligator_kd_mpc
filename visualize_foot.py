import numpy as np

import matplotlib.pyplot as plt

def load_points(file_path):
    # 假设每行数据为 x,y,z 格式，无标题
    return np.loadtxt(file_path, delimiter=',')

def main():
    # 读取数据
    fl_foot = load_points("/home/zishang/cpp_workspace/aligator_kd_mpc/build/fl_foot.csv")
    fl_foot_ref = load_points("/home/zishang/cpp_workspace/aligator_kd_mpc/build/fl_foot_ref.csv")
    # rr_foot = load_points("/home/zishang/cpp_workspace/aligator_kd_mpc/build/rr_foot.csv")
    # rr_foot_ref = load_points("/home/zishang/cpp_workspace/aligator_kd_mpc/build/rr_foot_ref.csv")

    # 设置三维绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制数据点
    ax.scatter(fl_foot[:, 0], fl_foot[:, 1], fl_foot[:, 2],
               c='red', marker='o', label='fl_foot')
    ax.scatter(fl_foot_ref[:, 0], fl_foot_ref[:, 1], fl_foot_ref[:, 2],
               c='blue', marker='^', label='fl_foot_ref')
    # ax.scatter(rr_foot[:, 0], rr_foot[:, 1], rr_foot[:, 2],
    #            c='green', marker='o', label='rr_foot')
    # ax.scatter(rr_foot_ref[:, 0], rr_foot_ref[:, 1], rr_foot_ref[:, 2],
    #            c='purple', marker='^', label='rr_foot_ref')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Foot Trajectory')
    ax.legend()
    
    plt.show()

if __name__ == '__main__':
    main()