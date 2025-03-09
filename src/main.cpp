#include "mpc.hpp"
#include <pinocchio/parsers/urdf.hpp>

using namespace simple_mpc;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

int main(int argc, char const *argv[])
{
    std::string urdf_file = "/home/zishang/Cpp_workspace/aligator_kd_mpc/robot/galileo_mini/robot.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_file, model);

    VectorXd reference_configuration(model.nq);
    reference_configuration << 0, 0, 0.38, 0, 0, 0, 1,
        0, 0.72, -1.44,
        0, 0.72, -1.44,
        0, 0.72, -1.44,
        0, 0.72, -1.44;
    std::string base_frame_name = "base_link";
    RobotModelHandler model_handler(model, reference_configuration, base_frame_name);
    model_handler.addFoot("FL_foot_link", base_frame_name,
                          SE3(Quaterniond::Identity(), Vector3d(0.3015, 0.1865, 0)));
    model_handler.addFoot("FR_foot_link", base_frame_name,
                          SE3(Quaterniond::Identity(), Vector3d(0.3015, -0.1865, 0)));
    model_handler.addFoot("HL_foot_link", base_frame_name,
                          SE3(Quaterniond::Identity(), Vector3d(-0.3015, 0.1865, 0)));
    model_handler.addFoot("HR_foot_link", base_frame_name,
                          SE3(Quaterniond::Identity(), Vector3d(-0.3015, -0.1865, 0)));

    int force_size = 3;
    int nk = model_handler.getFeetNames().size();
    Vector3d gravity(0, 0, -9.81);
    VectorXd f_ref = VectorXd::Zero(force_size);
    f_ref(2) = -model_handler.getMass() / nk * gravity(2);
    VectorXd u0(4 * force_size + model_handler.getModel().nv - 6);
    u0 << f_ref, f_ref, f_ref, f_ref, VectorXd::Zero(model_handler.getModel().nv - 6);
    
    return 0;
}
