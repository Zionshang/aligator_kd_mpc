import numpy as np
import pinocchio as pin
import pandas as pd
import time
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data as erd

csv_path = (
    "/home/zishang/cpp_workspace/aligator_kd_mpc/build/mpc_kinodynamics_result.csv"
)

# # 读取URDF文件和创建机器人模型
# urdf_path = "/home/zishang/cpp_workspace/aligator_kd_mpc/robot/galileo_mini/urdf/galileo_mini.urdf"
# model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
# visual_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL)
# collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION)

robot_wrapper = erd.load('go2')
model = robot_wrapper.model
visual_model = robot_wrapper.visual_model
collision_model = robot_wrapper.collision_model

# 设置可视化器
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(loadModel=True)
viz.viewer.open()

# 读取轨迹数据
try:
    trajectory_data = pd.read_csv(csv_path, header=None)
    x_trajectory = trajectory_data.values
    q_trajectory = x_trajectory[:, : model.nq]

    while True:
        for q in q_trajectory:
            viz.display(q)
            time.sleep(0.01)  # 可调整显示速度
        time.sleep(1)  # 可调整显示速度


except FileNotFoundError:
    print("找不到轨迹文件：trajectory_results.csv")
except Exception as e:
    print(f"发生错误：{str(e)}")