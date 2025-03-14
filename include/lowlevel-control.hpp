///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
/**
 * @file lowlevel-control.hpp
 * @brief Build a low-level control for kinodynamics
 * and centroidal MPC schemes
 */

#ifndef SIMPLE_MPC_LOWLEVEL_CONTROL_HPP_
#define SIMPLE_MPC_LOWLEVEL_CONTROL_HPP_

#include "fwd.hpp"
#include <pinocchio/multibody/fwd.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>

namespace simple_mpc
{
  using namespace proxsuite;
  using Eigen::Matrix3d;
  using Eigen::Vector3d;

  struct IDSettings
  {
    std::vector<FrameIndex> contact_ids; //< Index of contacts
    double mu;                           //< Friction parameter
    double Lfoot;                        //< Half-length of foot (if contact 6D)
    double Wfoot;                        //< Half-width of foot (if contact 6D)
    long force_size;                     //< Dimension of contact forces
    double kd;                           //< Baumgarte coefficient
    double w_acc;                        //< Weight for acceleration regularization
    double w_tau;                        //< Weight for torque regularization
    double w_force;                      //< Weight for force regularization
    bool verbose;                        //< Print solver information

    Matrix3d kp_sw; //< Proportional gains for swing foot tracking
    Matrix3d kd_sw; //< Derivative gains for swing foot tracking
  };

  class IDSolver
  {
  public:
    proxqp::dense::QP<double> qp_;

  protected:
    IDSettings settings_;
    pin::Model model_;
    pin::Data data_ref_;

    int force_dim_;
    int nforcein_;
    int nk_;

    Eigen::MatrixXd H_;
    Eigen::MatrixXd A_;
    Eigen::MatrixXd C_;
    Eigen::MatrixXd S_;
    Eigen::MatrixXd Cmin_;
    Eigen::VectorXd b_;
    Eigen::VectorXd g_;
    Eigen::VectorXd l_;
    Eigen::VectorXd u_;

    Eigen::MatrixXd J_;
    Eigen::VectorXd Jdot_qdot_;
    Eigen::MatrixXd Jdot_;
    Eigen::VectorXd a_feet_;

    // Internal matrix computation
    void computeMatrices(
        pin::Data &data,
        const std::vector<bool> &contact_state,
        const ConstVectorRef &v,
        const ConstVectorRef &q_ref,
        const ConstVectorRef &v_ref,
        const ConstVectorRef &a_ref,
        const ConstVectorRef &tau,
        const ConstVectorRef &forces,
        const ConstMatrixRef &M);

  public:
    explicit IDSolver(const IDSettings &settings, const pin::Model &model);

    void solveQP(
        pin::Data &data,
        const std::vector<bool> &contact_state,
        const ConstVectorRef &v,
        const ConstVectorRef &q_ref,
        const ConstVectorRef &v_ref,
        const ConstVectorRef &a_ref,
        const ConstVectorRef &tau,
        const ConstVectorRef &forces,
        const ConstMatrixRef &M);

    [[deprecated]] proxqp::dense::Model<double> getQP()
    {
      return qp_.model;
    }

    Eigen::MatrixXd getA()
    {
      return qp_.model.A;
    }
    Eigen::MatrixXd getH()
    {
      return qp_.model.H;
    }
    Eigen::MatrixXd getC()
    {
      return qp_.model.C;
    }
    Eigen::VectorXd getg()
    {
      return qp_.model.g;
    }
    Eigen::VectorXd getb()
    {
      return qp_.model.b;
    }

    // QP results
    Eigen::VectorXd solved_forces_;
    Eigen::VectorXd solved_acc_;
    Eigen::VectorXd solved_torque_;
  };

} // namespace simple_mpc

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

#endif // SIMPLE_MPC_LOWLEVEL_CONTROL_HPP_
