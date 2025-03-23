#include "mpc.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include "kinodynamics.hpp"
#include "lowlevel-control.hpp"
#include "webots_interface.hpp"
#include "utils/logger.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include "wbc/weighted_wbc.hpp"

using namespace simple_mpc;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

#define EXAMPLE_ROBOT_DATA_MODEL_DIR "/opt/openrobots/share/example-robot-data/robots"

int main(int argc, char const *argv[])
{
    // Load pinocchio model from example robot data
    Model model;
    std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/srdf/go2.srdf";

    pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
    pinocchio::srdf::loadReferenceConfigurations(model, srdf_path, false);

    std::string base_joint_name = "root_joint";
    RobotModelHandler model_handler(model, "standing", base_joint_name);
    model_handler.addFoot("FL_foot", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(0.17, 0.15, 0.0)));
    model_handler.addFoot("FR_foot", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(0.17, -0.15, 0.0)));
    model_handler.addFoot("RL_foot", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(-0.24, 0.15, 0.0)));
    model_handler.addFoot("RR_foot", base_joint_name, SE3(Quaterniond::Identity(), Vector3d(-0.24, -0.15, 0.0)));

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

    KinodynamicsSettings kd_settings;
    kd_settings.timestep = 0.01;
    kd_settings.w_x = w_x_vec.asDiagonal();
    kd_settings.w_u = w_u_vec.asDiagonal();
    kd_settings.w_frame = w_frame_vec.asDiagonal();
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
    mpc_settings.swing_apex = 0.30;
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
        {"FL_foot", true},
        {"FR_foot", true},
        {"RL_foot", true},
        {"RR_foot", true},
    };

    std::map<std::string, bool> contact_phase_lift_FL = {
        {"FL_foot", false},
        {"FR_foot", true},
        {"RL_foot", true},
        {"RR_foot", false},
    };

    std::map<std::string, bool> contact_phase_lift_FR = {
        {"FL_foot", true},
        {"FR_foot", false},
        {"RL_foot", false},
        {"RR_foot", true},
    };

    std::map<std::string, bool> contact_phase_lift = {
        {"FL_foot", false},
        {"FR_foot", false},
        {"RL_foot", false},
        {"RR_foot", false},
    };
    std::vector<std::map<std::string, bool>> contact_phases;
    contact_phases.insert(contact_phases.end(), T_ds, contact_phase_quadru);
    contact_phases.insert(contact_phases.end(), T_ss, contact_phase_lift_FL);
    contact_phases.insert(contact_phases.end(), T_ds, contact_phase_quadru);
    contact_phases.insert(contact_phases.end(), T_ss, contact_phase_lift_FR);
    // contact_phases.insert(contact_phases.end(), T, contact_phase_quadru);
    mpc.generateCycleHorizon(contact_phases);

    ////////////////////// 定义IDSolver //////////////////////
    IDSettings id_settings;
    id_settings.contact_ids = model_handler.getFeetIds();
    id_settings.mu = kd_settings.mu;
    id_settings.Lfoot = kd_settings.Lfoot;
    id_settings.Wfoot = kd_settings.Wfoot;
    id_settings.force_size = kd_settings.force_size;
    id_settings.w_acc = 1;
    id_settings.w_tau = 0;
    id_settings.w_force = 100;
    id_settings.verbose = false,
    id_settings.kp_sw = 350 * Matrix3d::Identity();
    id_settings.kd_sw = 37 * Matrix3d::Identity();
    IDSolver qp(id_settings, model_handler.getModel());

    ////////////////////// 定义Weighted WBC //////////////////////
    WbcSettings wbc_settings;
    wbc_settings.contact_ids = model_handler.getFeetIds();
    wbc_settings.mu = kd_settings.mu;
    wbc_settings.force_size = kd_settings.force_size;
    wbc_settings.kp_sw = 350 * Matrix3d::Identity();
    wbc_settings.kd_sw = 37 * Matrix3d::Identity();
    wbc_settings.w_swing = 100;
    wbc_settings.w_base = 1;
    wbc_settings.w_force = 0.01;
    WeightedWbc wbc(model_handler.getModel(), wbc_settings);

    ////////////////////// 设置仿真 //////////////////////
    int N_simu = 10;
    VectorXd v(6);
    v << 0.8, 0, 0, 0, 0, 0;
    mpc.velocity_base_ = v;
    VectorXd x_measure = model_handler.getReferenceState();
    WebotsInterface webots;
    int itr = 0;
    VectorXd a0, a1, forces0, forces1, q0, q1, v0, v1;
    std::vector<bool> contact_states;
    std::vector<VectorXd> x_logger, lf_foot_ref_logger, lf_foot_logger;

    while (webots.isRunning())
    {
        webots.recvState(x_measure);
        // mpc.switchToStand();
        if (int(itr % 10) == 0)
        {
            auto land_LF = mpc.getFootLandCycle("FL_foot");
            auto land_RF = mpc.getFootLandCycle("RL_foot");
            // std::cout << "landing_RF = " << land_RF << ", landing_LF = " << land_LF << std::endl;
            mpc.iterate(x_measure);
            a0 = mpc.getStateDerivative(0).tail(model.nv);
            a1 = mpc.getStateDerivative(1).tail(model.nv);
            a0.tail(12) = mpc.us_[0].tail(12);
            a1.tail(12) = mpc.us_[1].tail(12);
            q0 = mpc.xs_[0].head(model.nq);
            q1 = mpc.xs_[1].head(model.nq);
            v0 = mpc.xs_[0].tail(model.nv);
            v1 = mpc.xs_[1].tail(model.nv);
            forces0 = mpc.us_[0].head(nk * force_size);
            forces1 = mpc.us_[1].head(nk * force_size);
            contact_states = mpc.ocp_handler_->getContactState(0);
            itr = 0;
            // std::cout << "force0: " << forces0.transpose() << std::endl;
            // std::cout << "force1: " << forces1.transpose() << std::endl;
            std::cout << "Contact states: ";
            for (const auto &state : contact_states)
                std::cout << state << " ";
            std::cout << std::endl;
        }
        VectorXd a_interp = (double(N_simu) - itr) / double(N_simu) * a0 + itr / double(N_simu) * a1;
        VectorXd f_interp = (double(N_simu) - itr) / double(N_simu) * forces0 + itr / double(N_simu) * forces1;
        VectorXd q_interp = (double(N_simu) - itr) / double(N_simu) * q0 + itr / double(N_simu) * q1;
        VectorXd v_interp = (double(N_simu) - itr) / double(N_simu) * v0 + itr / double(N_simu) * v1;

        ////////////////////// 松弛WBC //////////////////////
        mpc.getDataHandler().updateInternalData(x_measure, true);
        qp.solveQP(
            mpc.getDataHandler().getData(),
            contact_states,
            x_measure.tail(model.nv),
            q_interp,
            v_interp,
            a_interp,
            VectorXd::Zero(12),
            f_interp,
            mpc.getDataHandler().getData().M);
        webots.sendCmd(qp.solved_torque_);

        ////////////////////// 加权WBC //////////////////////
        // VectorXd sol = wbc.update(x_measure.head(model.nq),
        //                           x_measure.tail(model.nv),
        //                           contact_states,
        //                           q_interp,
        //                           v_interp,
        //                           a_interp,
        //                           f_interp);
        // webots.sendCmd(sol.tail(12));

        itr++;

        // lf_foot_ref_logger.push_back(mpc.getReferencePose(0, "FL_foot").translation());
        // pinocchio::forwardKinematics(model_handler.getModel(), mpc.getDataHandler().getData(), x_measure.head(model_handler.getModel().nq));
        // pinocchio::updateFramePlacements(model_handler.getModel(), mpc.getDataHandler().getData());
        // lf_foot_logger.push_back(mpc.getDataHandler().getData().oMf[model_handler.getFootId("FL_foot")].translation());
        std::cout << "--------------------------" << std::endl;
    }

    // ////////////////////// 理想仿真 //////////////////////
    // VectorXd v(6);
    // v << 0.5, 0, 0, 0, 0, 0;
    // mpc.velocity_base_ = v;
    // VectorXd x_measure = model_handler.getReferenceState();
    // double sim_time = mpc_settings.timestep * T * 10;
    // std::vector<VectorXd> x_logger, lf_foot_ref_logger, lf_foot_logger;

    // for (int i = 0; i < int(sim_time / mpc_settings.timestep); i++)
    // {
    //     mpc.iterate(x_measure);
    //     std::cout << "i: " << i << " FL_foot ref pose: " << mpc.getReferencePose(0, "FL_foot").translation().transpose() << std::endl;
    //     x_measure = mpc.xs_[1];
    //     x_logger.push_back(x_measure);
    //     lf_foot_ref_logger.push_back(mpc.getReferencePose(0, "FL_foot").translation());
    //     pinocchio::forwardKinematics(model_handler.getModel(), mpc.getDataHandler().getData(), x_measure.head(model_handler.getModel().nq));
    //     pinocchio::updateFramePlacements(model_handler.getModel(), mpc.getDataHandler().getData());
    //     lf_foot_logger.push_back(mpc.getDataHandler().getData().oMf[model_handler.getFootId("FL_foot")].translation());
    // }

    // saveVectorsToCsv("mpc_kinodynamics_result.csv", x_logger);
    saveVectorsToCsv("lf_foot_ref_logger.csv", lf_foot_ref_logger);
    saveVectorsToCsv("lf_foot_logger.csv", lf_foot_logger);

    return 0;
}