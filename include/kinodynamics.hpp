///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "fwd.hpp"
#include "robot-handler.hpp"

#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/function-xpr-slice.hpp>
#include <proxsuite-nlp/modelling/constraints/box-constraint.hpp>
#include <proxsuite-nlp/modelling/constraints/negative-orthant.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

namespace simple_mpc
{
  using namespace aligator;
  using CostStack = CostStackTpl<double>;
  using ControlErrorResidual = ControlErrorResidualTpl<double>;
  using QuadraticControlCost = QuadraticControlCostTpl<double>;
  using QuadraticStateCost = QuadraticStateCostTpl<double>;
  using QuadraticResidualCost = QuadraticResidualCostTpl<double>;
  using StateErrorResidual = StateErrorResidualTpl<double>;
  using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;
  using NegativeOrthant = proxsuite::nlp::NegativeOrthantTpl<double>;
  using EqualityConstraint = proxsuite::nlp::EqualityConstraintTpl<double>;
  using FunctionSliceXpr = FunctionSliceXprTpl<double>;
  using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;

#define SIMPLE_MPC_DEFINE_DEFAULT_MOVE_CTORS(Type) \
  Type(Type &&) = default;                         \
  Type &operator=(Type &&) = default

  /**
   * @brief Build a kinodynamics problem based on
   * aligator's KinodynamicsFwdDynamics class.
   *
   * State is defined as concatenation of joint positions and
   * joint velocities; control is defined as concatenation of
   * contact forces and joint acceleration.
   */

  struct KinodynamicsSettings
  {
    /// timestep in problem shooting nodes
    double timestep;

    // Cost function weights
    Eigen::MatrixXd w_x;     // State
    Eigen::MatrixXd w_u;     // Control
    Eigen::MatrixXd w_frame; // End effector placement

    // Kinematics limits
    Eigen::VectorXd qmin;
    Eigen::VectorXd qmax;

    // Physics parameters
    Eigen::Vector3d gravity;
    double mu;
    int force_size;

    // Constraint
    bool kinematics_limits;
    bool force_cone;
  };

  class KinodynamicsOCP
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    explicit KinodynamicsOCP(const KinodynamicsSettings &settings, const RobotModelHandler &model_handler);

    SIMPLE_MPC_DEFINE_DEFAULT_MOVE_CTORS(KinodynamicsOCP);

    // Create one instance of Kinodynamics stage from desired contacts and forces
    StageModel createStage(const std::map<std::string, bool> &contact_phase,
                           const std::map<std::string, pinocchio::SE3> &contact_pose,
                           const std::map<std::string, Eigen::VectorXd> &contact_force);

    // Manage terminal cost
    CostStack createTerminalCost();

    void setReferenceFootPose(const std::size_t t, const std::string &ee_name, const Vector3d &pos_ref);
    void setReferenceState(const std::size_t t, const ConstVectorRef &x_ref);
    void setTerminalReferenceState(const ConstVectorRef &x_ref);
    void setReferenceContact(const std::size_t t, const Vector4i &contact);
    void setReferenceFootForce(const std::size_t t, const std::vector<VectorXd> &force_refs);

    const Eigen::VectorXd getReferenceForce(const std::size_t i, const std::string &cost_name);

    const Eigen::VectorXd getProblemState(const RobotDataHandler &data_handler);
    size_t getContactSupport(const std::size_t t);
    std::vector<bool> getContactState(const std::size_t t);

    void computeControlFromForces(const std::map<std::string, Eigen::VectorXd> &force_refs);

    // Create one TrajOptProblem from contact sequence
    void createProblem(const ConstVectorRef &x0,
                       const size_t horizon,
                       const int force_size,
                       const double gravity);

    void setReferenceControl(const std::size_t t, const ConstVectorRef &u_ref);
    ConstVectorRef getReferenceControl(const std::size_t t);
    ConstVectorRef getReferenceState(const std::size_t t);
    CostStack *getCostStack(std::size_t t);
    CostStack *getTerminalCostStack();
    int getConstraintSize(const std::size_t t);
    const pinocchio::SE3 getReferenceFootPose(const std::size_t i, const std::string &cost_name);

    std::size_t getSize() const
    {
      return problem_->numSteps();
    }

    TrajOptProblem &getProblem()
    {
      assert(problem_);
      return *problem_;
    }

    const TrajOptProblem &getProblem() const
    {
      assert(problem_);
      return *problem_;
    }

    const RobotModelHandler &getModelHandler() const
    {
      return model_handler_;
    }

  protected:
    KinodynamicsSettings settings_;
    Eigen::VectorXd x0_;

    // Size of the problem
    int nq_;
    int nv_;
    int ndx_;
    int nu_;

    /// The robot model
    RobotModelHandler model_handler_;
    pin::Data data_;
    
    /// The reference shooting problem storing all shooting nodes
    std::unique_ptr<TrajOptProblem> problem_;

    // Vector reference for control cost
    Eigen::VectorXd control_ref_;

    // Multibody phase space
    MultibodyPhaseSpace space_;
  };

} // namespace simple_mpc
