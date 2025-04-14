///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <aligator/core/stage-data.hpp>
#include <aligator/fwd.hpp>
#include <aligator/modelling/dynamics/fwd.hpp>
#include <aligator/modelling/dynamics/integrator-explicit.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include "planner/foot-trajectory.hpp"
#include "fwd.hpp"
#include "ocp.hpp"
#include "robot-handler.hpp"

namespace simple_mpc
{
  using ExplicitIntegratorData = dynamics::ExplicitIntegratorDataTpl<double>;

  struct MPCSettings
  {
  public:
    // Step-related quantities
    double swing_apex = 0.15;

    // Force parameters
    double support_force = 1000;

    // Solver-related quantities
    double TOL = 1e-4;
    double mu_init = 1e-8;
    std::size_t max_iters = 1;
    std::size_t num_threads = 2;
    int ddpIteration = 1;

    // Timings
    int T_fly = 80;
    int T_contact = 20;
    size_t T = 100; // todo: 暂时没有用？
    double timestep = 0.01;
  };

  /**
   * @brief Build a MPC object holding an instance
   * of a trajectory optimization problem
   */
  class MPC
  {

  protected:
    enum LocomotionType
    {
      WALKING,
      STANDING,
      MOTION
    };

    void updatePinocchioInfo(const VectorXd &q, const VectorXd &v);

    int nq_;
    int nv_;
    pin::Data data_;
    // contact_states_指的是cycle_horizon_中每个时刻的接触状态，而非MPC中的接触状态
    // MPC中的接触状态要比cycle_horizon_多一个一开始的完整stance周期
    std::vector<std::map<std::string, bool>> contact_states_;
    std::vector<std::shared_ptr<StageModel>> cycle_horizon_;  // 大小等于步态周期，而非MPC周期
    std::vector<std::shared_ptr<StageData>> cycle_horizon_data_;
    std::vector<std::shared_ptr<StageModel>> standing_horizon_;
    std::vector<std::shared_ptr<StageData>> standing_horizon_data_;
    FootTrajectory foot_trajectories_;
    std::map<std::string, pinocchio::SE3> relative_feet_poses_;
    // INTERNAL UPDATING function
    void updateStepTrackerReferences();

    // Memory preallocations:
    std::vector<unsigned long> controlled_joints_id_;
    std::vector<std::string> ee_names_;
    LocomotionType now_;

    std::shared_ptr<RobotDataHandler> data_handler_;

  public:
    std::unique_ptr<SolverProxDDP> solver_;
    Vector6d velocity_base_;
    Vector7d pose_base_;
    Eigen::Vector3d next_pose_;
    Eigen::Vector2d twist_vect_;
    MPCSettings settings_;
    std::shared_ptr<OCP> ocp_;

    explicit MPC(const MPCSettings &settings, std::shared_ptr<OCP> problem);

    // Generate the cycle walking problem along which we will iterate
    // the receding horizon
    void generateCycleHorizon(const std::vector<std::map<std::string, bool>> &contact_states);

    // Perform one iteration of MPC
    void iterate(const ConstVectorRef &x, double current_time);

    void updateCycleTiming(const bool updateOnlyHorizon);

    // Recede the horizon
    void recedeOnceCycle();

    void recedeCycles(double current_time);

    void setReferenceState(const std::vector<VectorXd> &x_ref) { x_ref_ = x_ref; }

    // getters and setters
    TrajOptProblem &getTrajOptProblem();

    RobotDataHandler &getDataHandler()
    {
      return *data_handler_;
    }
    const RobotModelHandler &getModelHandler() const
    {
      return ocp_->getModelHandler();
    }

    const ConstVectorRef getStateDerivative(const std::size_t t)
    {
      ExplicitIntegratorData *int_data =
          dynamic_cast<ExplicitIntegratorData *>(&*solver_->workspace_.problem_data.stage_data[t]->dynamics_data);
      assert(int_data != nullptr);
      return int_data->continuous_data->xdot_;
    }

    void switchToWalk(const Vector6d &velocity_base);

    void switchToStand();

    void testCost();

    // Footstep timings for each end effector
    std::map<std::string, std::vector<int>> foot_land_times_;

    // Solution vectors for state and control
    std::vector<VectorXd> xs_;
    std::vector<VectorXd> us_;
    std::vector<VectorXd> as_;

    // Riccati gains
    std::vector<MatrixXd> Ks_;

    // Reference state
    std::vector<VectorXd> x_ref_;

    // Initial quantities
    VectorXd x0_;
    VectorXd u0_;
    double last_recede_time_ = 0.0;
  };

} // namespace simple_mpc
