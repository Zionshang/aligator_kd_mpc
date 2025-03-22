#pragma once

#include "wbc_base.hpp"

class WeightedWbc : public WbcBase
{
public:
    using WbcBase::WbcBase;

    VectorXd update(const VectorXd &q, const VectorXd &v, std::vector<bool> contact_state,
                    const VectorXd &q_ref, const VectorXd &v_ref,
                    const VectorXd &a_ref, const VectorXd &force_ref) override;

protected:
    virtual Task formulateConstraints();

    virtual Task formulateWeightedTasks(const VectorXd &q, const VectorXd &v,
                                        const VectorXd &q_ref, const VectorXd &v_ref,
                                        const VectorXd &a_ref, const VectorXd &force_ref);
};