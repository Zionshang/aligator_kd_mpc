import example_robot_data as erd

# ####### CONFIGURATION  ############
# Load robot
URDF_SUBPATH = "/go2_description/urdf/go2.urdf"
base_joint_name ="root_joint"
robot_wrapper = erd.load('go2')
model = robot_wrapper.model
print(model)
