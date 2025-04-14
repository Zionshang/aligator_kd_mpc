#include "mpc/mpc.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include "mpc/ocp.hpp"
#include "webots_interface.hpp"
#include "utils/logger.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include "wbc/weighted_wbc.hpp"
#include "wbc/relaxed_wbc.hpp"
#include "utils/yaml_loader.hpp"
#include "mpc/interpolator.hpp"

using namespace simple_mpc;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

#define EXAMPLE_ROBOT_DATA_MODEL_DIR "/opt/openrobots/share/example-robot-data/robots"
#define WEBOTS
#define LOGGING

int main(int argc, char const *argv[])
{
    // Load pinocchio model from example robot data
    Model model;
    // std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    // std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/srdf/go2.srdf";
    std::string urdf_path = "/home/zishang/cpp_workspace/aligator_kd_mpc/robot/galileo_mini/urdf/galileo_mini.urdf";
    std::string srdf_path = "/home/zishang/cpp_workspace/aligator_kd_mpc/robot/galileo_mini/srdf/galileo_mini.srdf";

    pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
    pinocchio::srdf::loadReferenceConfigurations(model, srdf_path, false);
    const int nq = model.nq;
    const int nv = model.nv;

    std::string base_joint_name = "root_joint";
    RobotModelHandler model_handler(model, "standing", base_joint_name);
    model_handler.addFoot("FL_foot_link", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(0.32, 0.18, 0.0)));
    model_handler.addFoot("FR_foot_link", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(0.32, -0.18, 0.0)));
    model_handler.addFoot("HL_foot_link", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(-0.32, 0.18, 0.0)));
    model_handler.addFoot("HR_foot_link", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(-0.32, -0.18, 0.0)));

    int force_size = 3;
    int nk = model_handler.getFeetNames().size();
    Vector3d gravity(0, 0, -9.81);
    VectorXd f_ref = VectorXd::Zero(force_size);
    f_ref(2) = -model_handler.getMass() / nk * gravity(2);
    VectorXd u0(4 * force_size + model_handler.getModel().nv - 6);
    u0 << f_ref, f_ref, f_ref, f_ref, VectorXd::Zero(model_handler.getModel().nv - 6);

    /////////////////////////////////////// 定义权重 ///////////////////////////////////////
    std::string yaml_path = "/home/zishang/cpp_workspace/aligator_kd_mpc/config/parm.yaml";
    YamlParams params(yaml_path);
    params.printParams();

    VectorXd w_x_vec(2 * model.nv);
    w_x_vec << params.w_basepos, params.w_legpos, params.w_legpos, params.w_legpos, params.w_legpos,
        params.w_basevel, params.w_legvel, params.w_legvel, params.w_legvel, params.w_legvel;

    VectorXd w_u_vec(4 * force_size + model_handler.getModel().nv - 6);
    w_u_vec << params.w_force, params.w_force, params.w_force, params.w_force,
        params.w_legacc, params.w_legacc, params.w_legacc, params.w_legacc;

    VectorXd w_legpos_vec(model.nv - 6);
    w_u_vec << params.w_legpos, params.w_legpos, params.w_legpos, params.w_legpos;
    VectorXd w_legvel_vec(model.nv - 6);
    w_u_vec << params.w_legvel, params.w_legvel, params.w_legvel, params.w_legvel;

    OcpSettings ocp_settings;
    ocp_settings.timestep = params.timestep;
    ocp_settings.w_x = w_x_vec.asDiagonal();
    ocp_settings.w_u = w_u_vec.asDiagonal();
    ocp_settings.w_frame = params.w_foot.asDiagonal();
    ocp_settings.qmin = model_handler.getModel().lowerPositionLimit.tail(12);
    ocp_settings.qmax = model_handler.getModel().upperPositionLimit.tail(12);
    ocp_settings.gravity = gravity;
    ocp_settings.mu = params.friction;
    ocp_settings.force_size = force_size;
    ocp_settings.kinematics_limits = true;
    ocp_settings.w_body_trans = params.w_basepos.head(3).asDiagonal();
    ocp_settings.w_body_rot = params.w_basepos.tail(3).asDiagonal();
    ocp_settings.w_body_vel = params.w_basevel.asDiagonal();
    ocp_settings.w_leg_pos = w_legpos_vec.asDiagonal();
    ocp_settings.w_leg_vel = w_legvel_vec.asDiagonal();

    int T = params.horizon;
    auto kd_problem = std::make_shared<OCP>(ocp_settings, model_handler);
    kd_problem->createProblem(model_handler.getReferenceState(), T, force_size, gravity(2));

    double time_fly = 0.6;
    int T_fly = time_fly / ocp_settings.timestep;

    std::cout << "T_fly: " << T_fly << std::endl;

    MPCSettings mpc_settings;
    mpc_settings.swing_apex = 0.50;
    mpc_settings.support_force = -model_handler.getMass() * gravity(2);
    mpc_settings.TOL = 1e-4;
    mpc_settings.mu_init = 1e-8;
    mpc_settings.max_iters = params.max_iter;
    mpc_settings.num_threads = params.num_threads;
    mpc_settings.T_fly = T_fly;
    mpc_settings.T_contact = 0;
    mpc_settings.timestep = ocp_settings.timestep;
    mpc_settings.T = T;
    MPC mpc(mpc_settings, kd_problem);

    ////////////////////// 定义步态 //////////////////////
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
    std::vector<std::map<std::string, bool>> contact_phases;
    contact_phases.insert(contact_phases.end(), T_fly, contact_phase_lift_FL);
    contact_phases.insert(contact_phases.end(), T_fly, contact_phase_lift_FR);
    mpc.generateCycleHorizon(contact_phases);

    ////////////////////// 定义松弛WBC //////////////////////
    RelaxedWbcSettings Rwbc_settings;
    Rwbc_settings.contact_ids = model_handler.getFeetIds();
    Rwbc_settings.mu = ocp_settings.mu;
    Rwbc_settings.force_size = ocp_settings.force_size;
    Rwbc_settings.w_acc = 1;
    Rwbc_settings.w_force = 10;
    Rwbc_settings.verbose = false;
    RelaxedWbc relaxed_wbc(Rwbc_settings, model_handler.getModel());

////////////////////// 设置仿真 //////////////////////
#ifdef WEBOTS
    Interpolator interpolator(model);

    int N_simu = 10;
    VectorXd x_measure = model_handler.getReferenceState();
    WebotsInterface webots;
    int itr = 0;
    std::vector<bool> contact_states;
    std::vector<VectorXd> x_logger, fl_foot_ref_logger, fl_foot_logger, rr_foot_ref_logger, rr_foot_logger;
    int itr_mpc = 0;

    std::vector<VectorXd> pos_ref(mpc_settings.T, x_measure.head(nq));
    std::vector<VectorXd> vel_ref(mpc_settings.T, x_measure.tail(nv));
    std::vector<VectorXd> x_ref(mpc_settings.T, x_measure);
    VectorXd pos_ref_start = x_measure.head(nq);
    Vector3d v_cmd;

    const double dt = 0.001; // Time step for integration

    while (webots.isRunning())
    {
        double current_time = webots.current_time();
        webots.recvUserCmd(v_cmd);
        vel_ref[0].head(3) = v_cmd;

        webots.recvState(x_measure);
        // 设置参考轨迹
        pin::integrate(model, pos_ref_start, vel_ref[0] * dt, pos_ref[0]);
        pos_ref_start = pos_ref[0];

        x_ref[0].head(6) = x_measure.head(6);
        x_ref[0].segment(nq, 6) = x_measure.segment(nq, 6);

        for (int i = 1; i < mpc_settings.T; i++)
        {
            vel_ref[i] = vel_ref[i - 1];
            pin::integrate(model, pos_ref[i - 1], vel_ref[i - 1] * ocp_settings.timestep, pos_ref[i]);
            x_ref[i].head(nq) = pos_ref[i];
            x_ref[i].tail(nv) = vel_ref[i];
        }

        // mpc.switchToStand();
        if (int(itr % 10) == 0)
        {
            std::cout << "x_ref[0] = " << x_ref[0].transpose() << std::endl;
            std::cout << "x_ref[end] = " << x_ref.back().transpose() << std::endl;

            mpc.setReferenceState(x_ref);

            auto start_time = std::chrono::high_resolution_clock::now();
            mpc.iterate(x_measure, current_time);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
            std::cout << "MPC iteration time: " << elapsed.count() << " ms" << std::endl;

            contact_states = mpc.ocp_->getContactState(0);
            itr = 0;
            itr_mpc++;

            // // Print contact states in a single line
            // std::cout << "Contact states: ";
            // for (size_t i = 0; i < contact_states.size(); ++i)
            // {
            //     std::cout << (contact_states[i] ? "1" : "0") << " ";
            // }
            // std::cout << std::endl;

            // // Print contact states in a single line

            // for (size_t i = 0; i < kd_problem->getSize(); i++)
            // {
            //     std::cout << mpc.ocp_->getContactState(i)[0] << " ";
            // }
            // std::cout << std::endl;

#ifdef LOGGING
            fl_foot_ref_logger.push_back(kd_problem->getReferenceFootPose(0, "FL_foot_link").translation());
            rr_foot_ref_logger.push_back(kd_problem->getReferenceFootPose(0, "HR_foot_link").translation());
            pinocchio::forwardKinematics(model_handler.getModel(), mpc.getDataHandler().getData(), x_measure.head(model_handler.getModel().nq));
            pinocchio::updateFramePlacements(model_handler.getModel(), mpc.getDataHandler().getData());
            fl_foot_logger.push_back(mpc.getDataHandler().getData().oMf[model_handler.getFootId("FL_foot_link")].translation());
            rr_foot_logger.push_back(mpc.getDataHandler().getData().oMf[model_handler.getFootId("HR_foot_link")].translation());

            // if (itr_mpc == 100)
            // {
            //     for (int i = 0; i < kd_problem->getSize(); i++)
            //     {
            //         fl_foot_ref_logger.push_back(kd_problem->getReferenceFootPose(i, "FL_foot_link").translation());
            //         pinocchio::forwardKinematics(model_handler.getModel(), mpc.getDataHandler().getData(), mpc.xs_[i + 1].head(model_handler.getModel().nq));
            //         pinocchio::updateFramePlacements(model_handler.getModel(), mpc.getDataHandler().getData());
            //         fl_foot_logger.push_back(mpc.getDataHandler().getData().oMf[model_handler.getFootId("FL_foot_link")].translation());
            //     }
            // }

#endif
            // mpc.testCost();
            std::cout << "--------------------------" << std::endl;
        }

        double delay = itr * dt;

        VectorXd acc_interp, u_interp;
        interpolator.interpolateLinear(delay, ocp_settings.timestep, mpc.as_, acc_interp);
        interpolator.interpolateLinear(delay, ocp_settings.timestep, mpc.us_, u_interp);

        ////////////////////// 松弛WBC //////////////////////
        relaxed_wbc.solveQP(contact_states,
                            x_measure.head(nq),
                            x_measure.tail(nv),
                            acc_interp,
                            VectorXd::Zero(12),
                            u_interp.head(nk * force_size));
        webots.sendCmd(relaxed_wbc.solved_torque_);

        itr++;
    }
#else
    VectorXd x_measure = model_handler.getReferenceState();
    int itr = 0;
    std::vector<bool> contact_states;
    std::vector<VectorXd> x_logger, rf_foot_ref_logger, rf_foot_logger;

    std::vector<VectorXd> pos_ref(mpc_settings.T, x_measure.head(nq));
    std::vector<VectorXd> vel_ref(mpc_settings.T, x_measure.tail(nv));
    std::vector<VectorXd> x_ref(mpc_settings.T, x_measure);
    VectorXd pos_ref_start = x_measure.head(nq);
    double vx = 0;
    vel_ref[0](0) = vx;

    while (itr < 1000)
    {
        pin::integrate(model, pos_ref_start, vel_ref[0] * 0.001, pos_ref[0]);
        pos_ref_start = pos_ref[0];

        x_ref[0].head(nq) = pos_ref[0];
        x_ref[0].tail(nv) = vel_ref[0];
        for (int i = 1; i < mpc_settings.T; i++)
        {
            vel_ref[i] = vel_ref[i - 1];
            pin::integrate(model, pos_ref[i - 1], vel_ref[i - 1] * 0.001, pos_ref[i]);
            x_ref[i].head(nq) = pos_ref[i];
            x_ref[i].tail(nv) = vel_ref[i];
        }

        mpc.setReferenceState(x_ref);
        mpc.iterate(x_measure);

        x_measure = mpc.xs_[1];

        x_logger.push_back(x_measure);
        rf_foot_ref_logger.push_back(kd_problem->getReferenceFootPose(0, "FR_foot_link").translation());
        pinocchio::forwardKinematics(model_handler.getModel(), mpc.getDataHandler().getData(), x_measure.head(model_handler.getModel().nq));
        pinocchio::updateFramePlacements(model_handler.getModel(), mpc.getDataHandler().getData());
        rf_foot_logger.push_back(mpc.getDataHandler().getData().oMf[model_handler.getFootId("FR_foot_link")].translation());

        itr++;
        std::cout << "itr = " << itr << std::endl;
    }
#endif
    saveVectorsToCsv("x.csv", x_logger);
    saveVectorsToCsv("fl_foot_ref.csv", fl_foot_ref_logger);
    saveVectorsToCsv("fl_foot.csv", fl_foot_logger);
    saveVectorsToCsv("rr_foot_ref.csv", rr_foot_ref_logger);
    saveVectorsToCsv("rr_foot.csv", rr_foot_logger);
    return 0;
}