///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "lowlevel-control.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <proxsuite/proxqp/settings.hpp>

namespace simple_mpc
{

  IDSolver::IDSolver(const IDSettings &settings, const pin::Model &model)
      : qp_(1, 1, 1), settings_(settings), model_(model)
  {
    data_ref_ = pin::Data(model_);

    // Set the dimension of the problem
    nk_ = (int)settings.contact_ids.size();
    force_dim_ = (int)settings.force_size * nk_;
    int n = 2 * model_.nv - 6 + force_dim_;
    int neq = model_.nv + force_dim_;
    if (settings.force_size == 6)
      nforcein_ = 9;
    else
      nforcein_ = 5;
    int nin = nforcein_ * nk_ + model_.nv - 6;

    // Initialize QP matrices
    A_ = Eigen::MatrixXd::Zero(neq, n);
    b_ = Eigen::VectorXd::Zero(neq);
    l_ = Eigen::VectorXd::Zero(nin);
    u_ = Eigen::VectorXd::Ones(nin) * 100000;
    C_ = Eigen::MatrixXd::Zero(nin, n);
    g_ = Eigen::VectorXd::Zero(n);
    H_ = Eigen::MatrixXd::Zero(n, n);
    H_.topLeftCorner(model_.nv, model_.nv).diagonal() = Eigen::VectorXd::Ones(model_.nv) * settings_.w_acc;
    H_.block(model_.nv, model_.nv, force_dim_, force_dim_).diagonal() =
        Eigen::VectorXd::Ones(force_dim_) * settings_.w_force;
    H_.bottomRightCorner(model_.nv - 6, model_.nv - 6).diagonal() =
        Eigen::VectorXd::Ones(model_.nv - 6) * settings_.w_tau;

    // Initialize torque selection matrix
    S_ = Eigen::MatrixXd::Zero(model_.nv, model_.nv - 6);
    S_.bottomRows(model_.nv - 6).diagonal().setOnes();

    // Initialize full feet stacked Jacobian
    J_ = Eigen::MatrixXd::Zero(force_dim_, model_.nv); // ?也包含摆动腿吗？

    // Initialize derivative of contact Jacobian
    Jdot_ = Eigen::MatrixXd::Zero(6, model_.nv);

    // Initialize product of Jdot and qdot
    Jdot_qdot_ = Eigen::VectorXd::Zero(force_dim_);

    // Initialize desired feet acceleration
    a_feet_ = Eigen::VectorXd::Zero(force_dim_);

    // Create the block matrix used for contact force cone
    Cmin_.resize(nforcein_, settings.force_size);
    if (settings.force_size == 3)
    {
      Cmin_ << -1, 0, settings.mu, 1, 0, settings.mu, 0, -1, settings.mu, 0, 1, settings.mu, 0, 0, 1;
    }
    else
    {
      Cmin_ << -1, 0, settings.mu, 0, 0, 0, 1, 0, settings.mu, 0, 0, 0, 0, -1, settings.mu, 0, 0, 0, 0, 1, settings.mu,
          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, settings.Wfoot, -1, 0, 0, 0, 0, settings.Wfoot, 1, 0, 0, 0, 0, settings.Lfoot,
          0, -1, 0, 0, 0, settings.Lfoot, 0, 1, 0;
    }
    for (long i = 0; i < nk_; i++)
    {
      C_.block(i * nforcein_, model_.nv + i * settings_.force_size, nforcein_, settings_.force_size) = Cmin_;
    }

    // Set the block matrix for torque limits
    C_.bottomRightCorner(model_.nv - 6, model_.nv - 6).diagonal() = Eigen::VectorXd::Ones(model_.nv - 6);

    // Set size of solutions
    solved_forces_.resize(force_dim_);
    solved_acc_.resize(model_.nv);
    solved_torque_.resize(model_.nv - 6);

    // Create and initialize the QP object
    qp_ =
        proxqp::dense::QP<double>(n, neq, nin, false, proxqp::HessianType::Dense, proxqp::DenseBackend::PrimalDualLDLT);
    qp_.settings.eps_abs = 1e-3;
    qp_.settings.eps_rel = 0.0;
    qp_.settings.primal_infeasibility_solving = true;
    qp_.settings.check_duality_gap = true;
    qp_.settings.verbose = settings.verbose;
    qp_.settings.max_iter = 10;
    qp_.settings.max_iter_in = 10;

    qp_.init(H_, g_, A_, b_, C_, l_, u_);
  }

  void IDSolver::computeMatrices(
      pin::Data &data,
      const std::vector<bool> &contact_state,
      const ConstVectorRef &v,
      const ConstVectorRef &q_ref,
      const ConstVectorRef &v_ref,
      const ConstVectorRef &a_ref,
      const ConstVectorRef &tau,
      const ConstVectorRef &forces,
      const ConstMatrixRef &M)
  {
    // Reset matrices
    J_.setZero();
    Jdot_qdot_.setZero();
    l_.head(nforcein_ * nk_).setZero();
    C_.block(0, 0, nforcein_ * nk_, model_.nv + force_dim_).setZero();

    // Update diff torque lower and upper limits
    l_.tail(model_.nv - 6) = -model_.effortLimit.tail(model_.nv - 6) - tau;
    u_.tail(model_.nv - 6) = model_.effortLimit.tail(model_.nv - 6) - tau;

    // Update the problem with respect to current set of contacts
    for (long i = 0; i < nk_; i++)
    {
      Jdot_.setZero();
      getFrameJacobianTimeVariation(model_, data, settings_.contact_ids[(size_t)i], pin::LOCAL_WORLD_ALIGNED, Jdot_);
      J_.middleRows(i * settings_.force_size, settings_.force_size) =
          getFrameJacobian(model_, data, settings_.contact_ids[(size_t)i], pin::LOCAL_WORLD_ALIGNED)
              .topRows(settings_.force_size);
      Jdot_qdot_.segment(i * settings_.force_size, settings_.force_size) = Jdot_.topRows(settings_.force_size) * v;
      if (contact_state[(size_t)i])
      {
        a_feet_.segment(i * settings_.force_size, settings_.force_size).setZero();

        // Friction cone inequality update
        l_.segment(i * nforcein_, 5) << forces[i * settings_.force_size] - forces[i * settings_.force_size + 2] * settings_.mu,
            -forces[i * settings_.force_size] - forces[i * settings_.force_size + 2] * settings_.mu,
            forces[i * settings_.force_size + 1] - forces[i * settings_.force_size + 2] * settings_.mu,
            -forces[i * settings_.force_size + 1] - forces[i * settings_.force_size + 2] * settings_.mu,
            -forces[i * settings_.force_size + 2];
        if (nforcein_ == 9)
        {
          l_.segment(i * nforcein_ + 5, 4)
              << forces[i * settings_.force_size + 3] - forces[i * settings_.force_size + 2] * settings_.Wfoot,
              -forces[i * settings_.force_size + 3] - forces[i * settings_.force_size + 2] * settings_.Wfoot,
              forces[i * settings_.force_size + 4] - forces[i * settings_.force_size + 2] * settings_.Lfoot,
              -forces[i * settings_.force_size + 4] - forces[i * settings_.force_size + 2] * settings_.Lfoot;
        }

        C_.block(i * nforcein_, model_.nv + i * settings_.force_size, nforcein_, settings_.force_size) = Cmin_;
      }
      else
      {
        pin::forwardKinematics(model_, data_ref_, q_ref, v_ref);
        pin::updateFramePlacements(model_, data_ref_);
        Vector3d pos_foot_ref = data_ref_.oMf[settings_.contact_ids[(size_t)i]].translation();
        Vector3d pos_foot = data.oMf[settings_.contact_ids[(size_t)i]].translation();
        Vector3d vel_foot_ref = pin::getFrameVelocity(model_, data_ref_, settings_.contact_ids[(size_t)i], pin::LOCAL_WORLD_ALIGNED).linear();
        Vector3d vel_foot = pin::getFrameVelocity(model_, data, settings_.contact_ids[(size_t)i], pin::LOCAL_WORLD_ALIGNED).linear();
        a_feet_.segment(i * settings_.force_size, settings_.force_size) =
            settings_.kp_sw * (pos_foot_ref - pos_foot) + settings_.kd_sw * (vel_foot_ref - vel_foot);
      }
      std::cout << "a_feet_:" << a_feet_.transpose() << std::endl;
    }

    // Update equality matrices
    // A = [M -J' -S;
    //      J  0   0]
    A_.topLeftCorner(model_.nv, model_.nv) = M;
    A_.block(0, model_.nv, model_.nv, force_dim_) = -J_.transpose();
    A_.topRightCorner(model_.nv, model_.nv - 6) = -S_;
    A_.bottomLeftCorner(force_dim_, model_.nv) = J_;

    // b = [-nle - M*a + J'*forces + S*tau;
    //      -dJdq - J*a  + a_feet]
    b_.head(model_.nv) = -data.nle - M * a_ref + J_.transpose() * forces + S_ * tau;
    b_.tail(force_dim_) = -Jdot_qdot_ - J_ * a_ref + a_feet_;
  }

  void IDSolver::solveQP(
      pin::Data &data,
      const std::vector<bool> &contact_state,
      const ConstVectorRef &v,
      const ConstVectorRef &q_ref,
      const ConstVectorRef &v_ref,
      const ConstVectorRef &a_ref,
      const ConstVectorRef &tau,
      const ConstVectorRef &forces,
      const ConstMatrixRef &M)
  {
    computeMatrices(data, contact_state, v, q_ref, v_ref, a_ref, tau, forces, M);
    qp_.update(H_, g_, A_, b_, C_, l_, u_, false);
    qp_.solve();

    solved_acc_ = a_ref + qp_.results.x.head(model_.nv);
    solved_forces_ = forces + qp_.results.x.segment(model_.nv, force_dim_);
    solved_torque_ = tau + qp_.results.x.tail(model_.nv - 6);
  }

} // namespace simple_mpc
