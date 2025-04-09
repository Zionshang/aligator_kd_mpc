#include "wbc/relaxed_wbc.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <proxsuite/proxqp/settings.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace simple_mpc
{

    RelaxedWbc::RelaxedWbc(const RelaxedWbcSettings &settings, const pin::Model &model)
        : qp_(1, 1, 1), settings_(settings), model_(model), data_(model), nv_(model.nv), nq_(model.nq)
    {

        // Set the dimension of the problem
        nk_ = (int)settings.contact_ids.size();
        force_dim_ = (int)settings.force_size * nk_;
        int n = 2 * nv_ - 6 + force_dim_;
        int neq = nv_ + force_dim_;
        nforcein_ = 5;
        int nin = nforcein_ * nk_ + nv_ - 6;

        // Initialize QP matrices
        A_ = Eigen::MatrixXd::Zero(neq, n);
        b_ = Eigen::VectorXd::Zero(neq);
        l_ = Eigen::VectorXd::Zero(nin);
        u_ = Eigen::VectorXd::Ones(nin) * 100000;
        C_ = Eigen::MatrixXd::Zero(nin, n);
        g_ = Eigen::VectorXd::Zero(n);
        H_ = Eigen::MatrixXd::Zero(n, n);
        H_.topLeftCorner(nv_, nv_).diagonal() = Eigen::VectorXd::Ones(nv_) * settings_.w_acc;
        H_.block(nv_, nv_, force_dim_, force_dim_).diagonal() =
            Eigen::VectorXd::Ones(force_dim_) * settings_.w_force;

        // Initialize torque selection matrix
        S_ = Eigen::MatrixXd::Zero(nv_, nv_ - 6);
        S_.bottomRows(nv_ - 6).diagonal().setOnes();

        // Initialize full contact Jacobian
        Jc_ = Eigen::MatrixXd::Zero(force_dim_, nv_);

        // Initialize derivative of contact Jacobian
        Jdot_ = Eigen::MatrixXd::Zero(6, nv_);

        // Initialize acceleration drift
        Jdot_v_ = Eigen::VectorXd::Zero(force_dim_);

        // Create the block matrix used for contact force cone
        Cmin_.resize(nforcein_, settings.force_size);
        Cmin_ << -1, 0, settings.mu, 1, 0, settings.mu, 0, -1, settings.mu, 0, 1, settings.mu, 0, 0, 1;

        for (long i = 0; i < nk_; i++)
        {
            C_.block(i * nforcein_, nv_ + i * settings_.force_size, nforcein_, settings_.force_size) = Cmin_;
        }

        // Set the block matrix for torque limits
        // C_.bottomRightCorner(nv_ - 6, nv_ - 6).diagonal() = Eigen::VectorXd::Ones(nv_ - 6);

        // Set size of solutions
        solved_forces_.resize(force_dim_);
        solved_acc_.resize(nv_);
        solved_torque_.resize(nv_ - 6);

        // Create and initialize the QP object
        qp_ = proxqp::dense::QP<double>(n, neq, nin, false, proxqp::HessianType::Dense, proxqp::DenseBackend::PrimalDualLDLT);
        qp_.settings.eps_abs = 1e-3;
        qp_.settings.eps_rel = 0.0;
        qp_.settings.primal_infeasibility_solving = true;
        qp_.settings.check_duality_gap = true;
        qp_.settings.verbose = settings.verbose;
        qp_.settings.max_iter = 10;
        qp_.settings.max_iter_in = 10;

        qp_.init(H_, g_, A_, b_, C_, l_, u_);
    }
    void RelaxedWbc::updatePinocchioData(const ConstVectorRef &q, const ConstVectorRef &v)
    {
        forwardKinematics(model_, data_, q);
        updateFramePlacements(model_, data_);
        computeJointJacobians(model_, data_);
        computeJointJacobiansTimeVariation(model_, data_, q, v);
        crba(model_, data_, q);
        data_.M.triangularView<Eigen::StrictlyLower>() = data_.M.transpose().triangularView<Eigen::StrictlyLower>();
        nonLinearEffects(model_, data_, q, v);
    }

    void RelaxedWbc::computeMatrices(const std::vector<bool> &contact_state,
                                     const ConstVectorRef &v,
                                     const ConstVectorRef &a,
                                     const ConstVectorRef &tau,
                                     const ConstVectorRef &forces)
    {
        // Reset matrices
        Jc_.setZero();
        Jdot_v_.setZero();
        l_.head(nforcein_ * nk_).setZero();
        C_.block(0, 0, nforcein_ * nk_, nv_ + force_dim_).setZero();

        // Update diff torque lower and upper limits
        l_.tail(nv_ - 6) = -model_.effortLimit.tail(nv_ - 6) - tau;
        u_.tail(nv_ - 6) = model_.effortLimit.tail(nv_ - 6) - tau;

        // Update the problem with respect to current set of contacts
        for (long i = 0; i < nk_; i++)
        {
            Jdot_.setZero();
            if (contact_state[(size_t)i])
            {
                getFrameJacobianTimeVariation(model_, data_, settings_.contact_ids[(size_t)i], pin::LOCAL_WORLD_ALIGNED, Jdot_);
                Jc_.middleRows(i * settings_.force_size, settings_.force_size) =
                    getFrameJacobian(model_, data_, settings_.contact_ids[(size_t)i], pin::LOCAL_WORLD_ALIGNED).topRows(settings_.force_size);
                Jdot_v_.segment(i * settings_.force_size, settings_.force_size) = Jdot_.topRows(settings_.force_size) * v;

                // Friction cone inequality update
                l_.segment(i * nforcein_, 5) << forces[i * settings_.force_size] - forces[i * settings_.force_size + 2] * settings_.mu,
                    -forces[i * settings_.force_size] - forces[i * settings_.force_size + 2] * settings_.mu,
                    forces[i * settings_.force_size + 1] - forces[i * settings_.force_size + 2] * settings_.mu,
                    -forces[i * settings_.force_size + 1] - forces[i * settings_.force_size + 2] * settings_.mu,
                    -forces[i * settings_.force_size + 2];
                C_.block(i * nforcein_, nv_ + i * settings_.force_size, nforcein_, settings_.force_size) = Cmin_;
            }
        }

        // Update equality matrices
        A_.topLeftCorner(nv_, nv_) = data_.M;
        A_.block(0, nv_, nv_, force_dim_) = -Jc_.transpose();
        A_.topRightCorner(nv_, nv_ - 6) = -S_;
        A_.bottomLeftCorner(force_dim_, nv_) = Jc_;

        b_.head(nv_) = -data_.nle - data_.M * a + Jc_.transpose() * forces + S_ * tau;
        b_.tail(force_dim_) = -Jdot_v_ - Jc_ * a;
    }

    void RelaxedWbc::solveQP(const std::vector<bool> &contact_state,
                             const ConstVectorRef &q,
                             const ConstVectorRef &v,
                             const ConstVectorRef &a,
                             const ConstVectorRef &tau,
                             const ConstVectorRef &forces)
    {
        updatePinocchioData(q, v);
        computeMatrices(contact_state, v, a, tau, forces);
        qp_.update(H_, g_, A_, b_, C_, l_, u_, false);
        qp_.solve();

        solved_acc_ = a + qp_.results.x.head(nv_);
        solved_forces_ = forces + qp_.results.x.segment(nv_, force_dim_);
        solved_torque_ = tau + qp_.results.x.tail(nv_ - 6);
    }
} // namespace simple_mpc
