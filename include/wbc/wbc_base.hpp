#pragma once
#include "fwd.hpp"
#include "task.hpp"
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>

// Decision Variables: x = [\dot u^T, F^T, \tau^T]^T
using namespace simple_mpc;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Eigen::Matrix3d;
using Eigen::Vector3d;

struct WbcSettings
{
    std::vector<FrameIndex> contact_ids; //< Index of contacts
    double mu;                           //< Friction parameter
    long force_size;                     //< Dimension of contact forces
    Matrix3d kp_sw;                      //< Proportional gains for swing foot tracking
    Matrix3d kd_sw;                      //< Derivative gains for swing foot tracking
    double w_swing;                      //< Weight for swing foot tracking
    double w_force;                      //< Weight for force regularization
    double w_base;                       //< Weight for base acceleration
};

class WbcBase
{
public:
    virtual ~WbcBase() = default;

    WbcBase(const pin::Model &model, const WbcSettings &wbc_settings);

    virtual VectorXd update(const VectorXd &q, const VectorXd &v, std::vector<bool> contact_state,
        const VectorXd &q_ref, const VectorXd &v_ref,
        const VectorXd &a_ref, const VectorXd &force_ref);

protected:
    void updateMeasured(const VectorXd &q, const VectorXd &v);

    void updateFeference(const VectorXd &q_ref, const VectorXd &v_ref);

    size_t getNumDecisionVars() const { return num_decision_vars_; }

    Task formulateFloatingBaseEomTask();

    Task formulateTorqueLimitsTask();

    Task formulateNoContactMotionTask();

    Task formulateFrictionConeTask();

    Task formulateBaseAccelTask(const VectorXd &a_ref);

    Task formulateSwingLegTask(const VectorXd &q_ref, const VectorXd &v_ref,
                               const VectorXd &q, const VectorXd &v);

    Task formulateContactForceTask(const VectorXd &force_ref) const;

    size_t num_decision_vars_;
    WbcSettings wbc_settings_;
    pin::Model model_;
    pin::Data data_, data_ref_;

    int nc_;
    MatrixXd j_, dj_;
    VectorXd djdq_;
    std::vector<bool> contact_state_;
    size_t num_contacts_;
};