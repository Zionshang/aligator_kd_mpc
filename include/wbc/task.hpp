#pragma once

#include <Eigen/Core>
#include <utility>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Task
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Task() = default;

    Task(MatrixXd a, VectorXd b, MatrixXd d, VectorXd f) : a_(std::move(a)), d_(std::move(d)), b_(std::move(b)),
                                                           f_(std::move(f))
    {
    }

    explicit Task(size_t numDecisionVars)
        : Task(MatrixXd::Zero(0, numDecisionVars), VectorXd::Zero(0), MatrixXd::Zero(0, numDecisionVars),
               VectorXd::Zero(0))
    {
    }

    Task operator+(const Task &rhs) const
    {
        return {
            concatenateMatrices(a_, rhs.a_), concatenateVectors(b_, rhs.b_), concatenateMatrices(d_, rhs.d_),
            concatenateVectors(f_, rhs.f_)};
    }

    Task operator*(double rhs) const
    {
        // clang-format off
            return {
                a_.cols() > 0 ? rhs * a_ : a_,
                b_.cols() > 0 ? rhs * b_ : b_,
                d_.cols() > 0 ? rhs * d_ : d_,
                f_.cols() > 0 ? rhs * f_ : f_
            }; // clang-format on
    }

    MatrixXd a_, d_;
    VectorXd b_, f_;

    static MatrixXd concatenateMatrices(MatrixXd m1, MatrixXd m2)
    {
        if (m1.cols() <= 0)
        {
            return m2;
        }
        if (m2.cols() <= 0)
        {
            return m1;
        }
        assert(m1.cols() == m2.cols());
        MatrixXd res(m1.rows() + m2.rows(), m1.cols());
        res << m1, m2;
        return res;
    }

    static VectorXd concatenateVectors(const VectorXd &v1, const VectorXd &v2)
    {
        if (v1.cols() <= 0)
        {
            return v2;
        }
        if (v2.cols() <= 0)
        {
            return v1;
        }
        assert(v1.cols() == v2.cols());
        VectorXd res(v1.rows() + v2.rows());
        res << v1, v2;
        return res;
    }
};
