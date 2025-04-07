#pragma once
#include "fwd.hpp"
#include <pinocchio/multibody/fwd.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>

namespace simple_mpc
{
    using namespace proxsuite;
    struct RelaxedWbcSettings
    {
        std::vector<FrameIndex> contact_ids; //< Index of contacts
        double mu;                           //< Friction parameter
        long force_size;                     //< Dimension of contact forces
        double w_acc;                        //< Weight for acceleration regularization
        double w_force;                      //< Weight for force regularization
        bool verbose;                        //< Print solver information
    };

    class RelaxedWbc
    {
    public:
        proxqp::dense::QP<double> qp_;

    protected:
        RelaxedWbcSettings settings_;
        pin::Model model_;
        pin::Data data_;

        int force_dim_;
        int nforcein_;
        int nk_;
        int nv_;
        int nq_;

        Eigen::MatrixXd H_;
        Eigen::MatrixXd A_;
        Eigen::MatrixXd C_;
        Eigen::MatrixXd S_;
        Eigen::MatrixXd Cmin_;
        Eigen::VectorXd b_;
        Eigen::VectorXd g_;
        Eigen::VectorXd l_;
        Eigen::VectorXd u_;

        Eigen::MatrixXd Jc_;
        Eigen::VectorXd Jdot_v_;
        Eigen::MatrixXd Jdot_;

        // Internal matrix computation
        void computeMatrices(const std::vector<bool> &contact_state,
                             const ConstVectorRef &v,
                             const ConstVectorRef &a,
                             const ConstVectorRef &tau,
                             const ConstVectorRef &forces,
                             const ConstMatrixRef &M);

        void updatePinocchioData(const ConstVectorRef &q, const ConstVectorRef &v);

    public:
        explicit RelaxedWbc(const RelaxedWbcSettings &settings, const pin::Model &model);

        void solveQP(const std::vector<bool> &contact_state,
                     const ConstVectorRef &q,
                     const ConstVectorRef &v,
                     const ConstVectorRef &a,
                     const ConstVectorRef &tau,
                     const ConstVectorRef &forces,
                     const ConstMatrixRef &M);

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

}
