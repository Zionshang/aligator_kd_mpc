#include "wbc/weighted_wbc.hpp"
#include <qpOASES.hpp>

VectorXd WeightedWbc::update(const VectorXd &q, const VectorXd &v, std::vector<bool> contact_state,
                             const VectorXd &q_ref, const VectorXd &v_ref,
                             const VectorXd &a_ref, const VectorXd &force_ref)
{
    WbcBase::update(q, v, contact_state, q_ref, v_ref, a_ref, force_ref);

    // Constraints
    Task constraints = formulateConstraints();
    size_t numConstraints = constraints.b_.size() + constraints.f_.size();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(numConstraints, getNumDecisionVars());
    VectorXd lbA(numConstraints), ubA(numConstraints); // clang-format off
        A << constraints.a_,
                constraints.d_;

        lbA << constraints.b_,
                -qpOASES::INFTY * VectorXd::Ones(constraints.f_.size());
        ubA << constraints.b_,
                constraints.f_; // clang-format on

    // Cost
    Task swing_task = formulateSwingLegTask(q_ref, v_ref, q, v);
    Task base_task = formulateBaseAccelTask(a_ref);
    Task force_task = formulateContactForceTask(force_ref);
    Task weighedTask = swing_task * wbc_settings_.w_swing + base_task * wbc_settings_.w_base + force_task * wbc_settings_.w_force;

    // Task weighedTask = formulateWeightedTasks(q, v, q_ref, v_ref, a_ref, force_ref);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> H =
        weighedTask.a_.transpose() * weighedTask.a_;
    VectorXd g = -weighedTask.a_.transpose() * weighedTask.b_;

    // Solve
    auto qpProblem = qpOASES::QProblem(getNumDecisionVars(), numConstraints);
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_LOW;
    options.enableEqualities = qpOASES::BT_TRUE;
    qpProblem.setOptions(options);
    int nWsr = 20;

    qpProblem.init(H.data(), g.data(), A.data(), nullptr, nullptr, lbA.data(), ubA.data(), nWsr);
    VectorXd qpSol(getNumDecisionVars());

    qpProblem.getPrimalSolution(qpSol.data());

    // // 验证task
    // auto swing_task_error = swing_task.a_ * qpSol - swing_task.b_;
    // auto base_task_error = base_task.a_ * qpSol - base_task.b_;
    // auto force_task_error = force_task.a_ * qpSol - force_task.b_;
    // std::cout << "swing_task_error: " << swing_task_error.transpose() << std::endl;
    // std::cout << "base_task_error: " << base_task_error.transpose() << std::endl;
    // std::cout << "force_task_error: " << force_task_error.transpose() << std::endl;
    return qpSol;
}

Task WeightedWbc::formulateConstraints()
{
    return formulateFloatingBaseEomTask() +
        //    formulateTorqueLimitsTask() +
           formulateFrictionConeTask() +
           formulateNoContactMotionTask();
}

Task WeightedWbc::formulateWeightedTasks(const VectorXd &q, const VectorXd &v,
                                         const VectorXd &q_ref, const VectorXd &v_ref,
                                         const VectorXd &a_ref, const VectorXd &force_ref)
{
    return formulateSwingLegTask(q_ref, v_ref, q, v) * wbc_settings_.w_swing +
           formulateBaseAccelTask(a_ref) * wbc_settings_.w_base +
           formulateContactForceTask(force_ref) * wbc_settings_.w_force;
}