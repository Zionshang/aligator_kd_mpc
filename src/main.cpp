#include "mpc.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include "kinodynamics.hpp"
#include "utils/project_path.hpp"
#include "utils/logger.hpp"
#include "webots_interface.hpp"

using namespace simple_mpc;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

#define EXAMPLE_ROBOT_DATA_MODEL_DIR "/opt/openrobots/share/example-robot-data/robots"

int main(int argc, char const *argv[])
{
    // Load pinocchio model from example robot data
    Model model;
    std::string urdf_path = getProjectPath() + "/robot/galileo_mini/urdf/galileo_mini.urdf";
    std::string srdf_path = getProjectPath() + "/robot/galileo_mini/srdf/galileo_mini.srdf";

    pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
    pinocchio::srdf::loadReferenceConfigurations(model, srdf_path, false);

    std::string base_joint_name = "root_joint";
    RobotModelHandler model_handler(model, "standing", base_joint_name);
    model_handler.addFoot("FL_foot_link", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(0.3015, 0.1865, 0)));
    model_handler.addFoot("FR_foot_link", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(0.3015, -0.1865, 0)));
    model_handler.addFoot("HL_foot_link", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(-0.3015, 0.1865, 0)));
    model_handler.addFoot("HR_foot_link", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(-0.3015, -0.1865, 0)));

    int force_size = 3;
    int nk = model_handler.getFeetNames().size();
    Vector3d gravity(0, 0, -9.81);
    VectorXd f_ref = VectorXd::Zero(force_size);
    f_ref(2) = -model_handler.getMass() / nk * gravity(2);
    VectorXd u0(4 * force_size + model_handler.getModel().nv - 6);
    u0 << f_ref, f_ref, f_ref, f_ref, VectorXd::Zero(model_handler.getModel().nv - 6);

    /////////////////////////////////////// 定义权重 ///////////////////////////////////////
    VectorXd w_basepos(6);
    w_basepos << 0, 0, 100, 10, 10, 0;
    VectorXd w_legpos(3);
    w_legpos << 1, 1, 1;
    VectorXd w_basevel(6);
    w_basevel << 10, 10, 10, 10, 10, 10;
    VectorXd w_legvel(3);
    w_legvel << 0.1, 0.1, 0.1;
    VectorXd w_x_vec(2 * model.nv);
    w_x_vec << w_basepos, w_legpos, w_legpos, w_legpos, w_legpos, w_basevel, w_legvel, w_legvel, w_legvel, w_legvel;

    VectorXd w_force(3);
    w_force << 0.01, 0.01, 0.01;
    VectorXd w_u_vec(4 * force_size + model_handler.getModel().nv - 6);
    w_u_vec << w_force, w_force, w_force, w_force, Eigen::VectorXd::Ones(model_handler.getModel().nv - 6) * 1e-5;

    VectorXd w_frame_vec(3);
    w_frame_vec << 2000, 2000, 2000;
    VectorXd w_cent_vec(6);
    w_cent_vec << 0.0, 0.0, 1.0, 0.1, 0.1, 10.0;
    VectorXd w_centder_vec(6);
    w_centder_vec << 0.0, 0.0, 0.0, 0.1, 0.1, 0.1;

    KinodynamicsSettings kd_settings;
    kd_settings.timestep = 0.01;
    kd_settings.w_x = w_x_vec.asDiagonal();
    kd_settings.w_u = w_u_vec.asDiagonal();
    kd_settings.w_frame = w_frame_vec.asDiagonal();
    kd_settings.w_cent = w_cent_vec.asDiagonal();
    kd_settings.w_centder = w_centder_vec.asDiagonal();
    kd_settings.qmin = model_handler.getModel().lowerPositionLimit.tail(12);
    kd_settings.qmax = model_handler.getModel().upperPositionLimit.tail(12);
    kd_settings.gravity = gravity;
    kd_settings.mu = 0.8;
    kd_settings.Lfoot = 0.01;
    kd_settings.Wfoot = 0.01;
    kd_settings.force_size = force_size;
    kd_settings.kinematics_limits = true;
    kd_settings.force_cone = true;

    int T = 50;
    auto kd_problem = std::make_shared<KinodynamicsOCP>(kd_settings, model_handler);
    kd_problem->createProblem(model_handler.getReferenceState(), T, force_size, gravity(2), false);

    int T_ds = 10;
    int T_ss = 30;

    MPCSettings mpc_settings;
    mpc_settings.swing_apex = 0.15;
    mpc_settings.support_force = -model_handler.getMass() * gravity(2);
    mpc_settings.TOL = 1e-4;
    mpc_settings.mu_init = 1e-8;
    mpc_settings.max_iters = 1;
    mpc_settings.num_threads = 1;
    mpc_settings.T_fly = T_ss;
    mpc_settings.T_contact = T_ds;
    mpc_settings.timestep = kd_settings.timestep;

    MPC mpc(mpc_settings, kd_problem);

    ////////////////////// 定义步态 //////////////////////
    std::map<std::string, bool> contact_phase_quadru = {
        {"FL_foot_link", true},
        {"FR_foot_link", true},
        {"HL_foot_link", true},
        {"HR_foot_link", true},
    };

    std::map<std::string, bool> contact_phase_lift_FL = {
        {"FL_foot_link", false},
        {"FR_foot_link", true},
        {"HL_foot_link", true},
        {"HR_foot_link", false},
    };

    std::map<std::string, bool> contact_phase_lift_FR = {
        {"FL_foot_link", true},
        {"FR_foot_link", false},
        {"HL_foot_link", false},
        {"HR_foot_link", true},
    };

    std::map<std::string, bool> contact_phase_lift = {
        {"FL_foot_link", false},
        {"FR_foot_link", false},
        {"HL_foot_link", false},
        {"HR_foot_link", false},
    };
    std::vector<std::map<std::string, bool>> contact_phases;
    contact_phases.insert(contact_phases.end(), T_ds, contact_phase_quadru);
    contact_phases.insert(contact_phases.end(), T_ss, contact_phase_lift_FL);
    contact_phases.insert(contact_phases.end(), T_ds, contact_phase_quadru);
    contact_phases.insert(contact_phases.end(), T_ss, contact_phase_lift_FR);

    mpc.generateCycleHorizon(contact_phases);

    Vector6d v_base;
    v_base << 1, 0, 0, 0, 0, 0;
    mpc.setVelocityBase(v_base);

    // VectorXd x_measured(model.nq + model.nv);
    // WebotsInterface webots_interface;
    // while (webots_interface.isRunning())
    // {
    //     webots_interface.recvState(x_measured);

    //     auto start = std::chrono::high_resolution_clock::now();
    //     mpc.iterate(x_measured);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double, std::milli> iter_time = end - start;
    //     std::cout << iter_time.count() << " ms" << std::endl;
    //     webots_interface.sendCmd(mpc.us_[0]);
    // }

    VectorXd x_measured = model_handler.getReferenceState();
    std::vector<VectorXd> x_logger;
    double sim_time = 0.05; // second;
    for (size_t i = 0; i < int(sim_time / mpc_settings.timestep); i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        mpc.iterate(x_measured);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> iter_time = end - start;
        std::cout << "Iteration " << i << " took " << iter_time.count() << " ms" << std::endl;
        x_measured = mpc.xs_[1];
        x_logger.push_back(x_measured);
    }
    saveVectorsToCsv("mpc_kinodynamics_result.csv", x_logger);

    return 0;
}
