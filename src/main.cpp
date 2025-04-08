#include "mpc.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include "kinodynamics.hpp"
#include "webots_interface.hpp"
#include "utils/logger.hpp"
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include "wbc/weighted_wbc.hpp"
#include "wbc/relaxed_wbc.hpp"

using namespace simple_mpc;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;

#define EXAMPLE_ROBOT_DATA_MODEL_DIR "/opt/openrobots/share/example-robot-data/robots"
#define WEBOTS

int main(int argc, char const *argv[])
{
    // Load pinocchio model from example robot data
    Model model;
    std::string urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/urdf/go2.urdf";
    std::string srdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/go2_description/srdf/go2.srdf";

    pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), model);
    pinocchio::srdf::loadReferenceConfigurations(model, srdf_path, false);
    const int nq = model.nq;
    const int nv = model.nv;

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
    w_basepos << 100, 100, 1500, 300, 300, 100;
    VectorXd w_legpos(3);
    w_legpos << 10, 10, 10;
    VectorXd w_basevel(6);
    w_basevel << 10, 10, 100, 30, 30, 10;
    VectorXd w_legvel(3);
    w_legvel << 0.1, 0.1, 0.1;
    VectorXd w_x_vec(2 * model.nv);
    w_x_vec << w_basepos, w_legpos, w_legpos, w_legpos, w_legpos, w_basevel, w_legvel, w_legvel, w_legvel, w_legvel;

    VectorXd w_force(3);
    w_force << 0.001, 0.001, 0.001;
    VectorXd w_u_vec(4 * force_size + model_handler.getModel().nv - 6);
    w_u_vec << w_force, w_force, w_force, w_force, Eigen::VectorXd::Ones(model_handler.getModel().nv - 6) * 1e-5;

    VectorXd w_frame_vec(3);
    w_frame_vec << 5000, 5000, 5000;

    KinodynamicsSettings kd_settings;
    kd_settings.timestep = 0.01;
    kd_settings.w_x = w_x_vec.asDiagonal();
    kd_settings.w_u = w_u_vec.asDiagonal();
    kd_settings.w_frame = w_frame_vec.asDiagonal();
    kd_settings.qmin = model_handler.getModel().lowerPositionLimit.tail(12);
    kd_settings.qmax = model_handler.getModel().upperPositionLimit.tail(12);
    kd_settings.gravity = gravity;
    kd_settings.mu = 0.8;
    kd_settings.force_size = force_size;
    kd_settings.kinematics_limits = true;
    kd_settings.force_cone = true;

    int T = 50;
    auto kd_problem = std::make_shared<KinodynamicsOCP>(kd_settings, model_handler);
    kd_problem->createProblem(model_handler.getReferenceState(), T, force_size, gravity(2));

    int T_ds = 0;
    int T_ss = 30;

    MPCSettings mpc_settings;
    mpc_settings.swing_apex = 0.50;
    mpc_settings.support_force = -model_handler.getMass() * gravity(2);
    mpc_settings.TOL = 1e-4;
    mpc_settings.mu_init = 1e-8;
    mpc_settings.max_iters = 1;
    mpc_settings.num_threads = 1;
    mpc_settings.T_fly = T_ss;
    mpc_settings.T_contact = T_ds;
    mpc_settings.timestep = kd_settings.timestep;
    mpc_settings.T = T;
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

    ////////////////////// 定义松弛WBC //////////////////////
    RelaxedWbcSettings Rwbc_settings;
    Rwbc_settings.contact_ids = model_handler.getFeetIds();
    Rwbc_settings.mu = kd_settings.mu;
    Rwbc_settings.force_size = kd_settings.force_size;
    Rwbc_settings.w_acc = 1;
    Rwbc_settings.w_force = 10;
    Rwbc_settings.verbose = false;
    RelaxedWbc relaxed_wbc(Rwbc_settings, model_handler.getModel());

////////////////////// 设置仿真 //////////////////////
#ifdef WEBOTS
    int N_simu = 10;
    VectorXd x_measure = model_handler.getReferenceState();
    WebotsInterface webots;
    int itr = 0;
    VectorXd a0, a1, forces0, forces1;
    std::vector<bool> contact_states;
    std::vector<VectorXd> x_logger, fl_foot_ref_logger, fl_foot_logger, rr_foot_ref_logger, rr_foot_logger;
    int itr_mpc = 0;

    std::vector<VectorXd> pos_ref(mpc_settings.T, x_measure.head(nq));
    std::vector<VectorXd> vel_ref(mpc_settings.T, x_measure.tail(nv));
    std::vector<VectorXd> x_ref(mpc_settings.T, x_measure);
    VectorXd pos_ref_start = x_measure.head(nq);

    const double dt = 0.001; // Time step for integration

    while (webots.isRunning())
    {
        if (itr_mpc > 100)
        {
            double vx = 0.4;
            vel_ref[0](0) = vx;
        }

        webots.recvState(x_measure);
        // 设置参考轨迹
        pin::integrate(model, pos_ref_start, vel_ref[0] * dt, pos_ref[0]);
        pos_ref_start = pos_ref[0];

        x_ref[0].head(nq) = pos_ref[0];
        x_ref[0].tail(nv) = vel_ref[0];
        for (int i = 1; i < mpc_settings.T; i++)
        {
            vel_ref[i] = vel_ref[i - 1];
            pin::integrate(model, pos_ref[i - 1], vel_ref[i - 1] * kd_settings.timestep, pos_ref[i]);
            x_ref[i].head(nq) = pos_ref[i];
            x_ref[i].tail(nv) = vel_ref[i];
        }

        // mpc.switchToStand();
        if (int(itr % 10) == 0)
        {
            std::cout << "itr_mpc = " << itr_mpc << std::endl;
            mpc.setReferenceState(x_ref);

            auto start_time = std::chrono::high_resolution_clock::now();
            mpc.iterate(x_measure);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
            std::cout << "MPC iteration time: " << elapsed.count() << " ms" << std::endl;

            a0 = mpc.getStateDerivative(0).tail(model.nv);
            a1 = mpc.getStateDerivative(1).tail(model.nv);
            a0.tail(12) = mpc.us_[0].tail(12); // ? 这里是否有必要？
            a1.tail(12) = mpc.us_[1].tail(12);
            forces0 = mpc.us_[0].head(nk * force_size);
            forces1 = mpc.us_[1].head(nk * force_size);
            contact_states = mpc.ocp_handler_->getContactState(0);
            itr = 0;
            itr_mpc++;

            fl_foot_ref_logger.push_back(kd_problem->getReferenceFootPose(0, "FL_foot").translation());
            rr_foot_ref_logger.push_back(kd_problem->getReferenceFootPose(0, "RR_foot").translation());
            pinocchio::forwardKinematics(model_handler.getModel(), mpc.getDataHandler().getData(), x_measure.head(model_handler.getModel().nq));
            pinocchio::updateFramePlacements(model_handler.getModel(), mpc.getDataHandler().getData());
            fl_foot_logger.push_back(mpc.getDataHandler().getData().oMf[model_handler.getFootId("FL_foot")].translation());
            rr_foot_logger.push_back(mpc.getDataHandler().getData().oMf[model_handler.getFootId("RR_foot")].translation());
            std::cout << "--------------------------" << std::endl;
        }
        VectorXd a_interp = (double(N_simu) - itr) / double(N_simu) * a0 + itr / double(N_simu) * a1;
        VectorXd f_interp = (double(N_simu) - itr) / double(N_simu) * forces0 + itr / double(N_simu) * forces1;

        ////////////////////// 松弛WBC //////////////////////
        relaxed_wbc.solveQP(contact_states,
                            x_measure.head(nq),
                            x_measure.tail(nv),
                            a_interp,
                            VectorXd::Zero(12),
                            f_interp,
                            mpc.getDataHandler().getData().M);
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
        rf_foot_ref_logger.push_back(kd_problem->getReferenceFootPose(0, "FR_foot").translation());
        pinocchio::forwardKinematics(model_handler.getModel(), mpc.getDataHandler().getData(), x_measure.head(model_handler.getModel().nq));
        pinocchio::updateFramePlacements(model_handler.getModel(), mpc.getDataHandler().getData());
        rf_foot_logger.push_back(mpc.getDataHandler().getData().oMf[model_handler.getFootId("FR_foot")].translation());

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