#include "wbc/wbc_base.hpp"

WbcBase::WbcBase(const pin::Model &model, const WbcSettings &wbc_settings)
    : model_(model), wbc_settings_(wbc_settings)
{
    nc_ = wbc_settings.contact_ids.size();
    num_decision_vars_ = model.nv + wbc_settings_.force_size * nc_ + model.nv - 6;

    data_ = pin::Data(model_);
    data_ref_ = pin::Data(model_);

    dj_.setZero(3 * wbc_settings_.force_size, model_.nv);
    j_.setZero(3 * 4, model_.nv);
}

VectorXd WbcBase::update(const VectorXd &q, const VectorXd &v, std::vector<bool> contact_state,
                         const VectorXd &q_ref, const VectorXd &v_ref,
                         const VectorXd &a_ref, const VectorXd &force_ref)
{
    contact_state_ = contact_state;
    num_contacts_ = 0;
    for (const auto &flag : contact_state_)
    {
        if (flag)
            num_contacts_++;
    }
    updateMeasured(q, v);
    updateFeference(q_ref, v_ref);

    return {};
}

void WbcBase::updateMeasured(const VectorXd &q, const VectorXd &v)
{
    // For floating base EoM task
    pin::forwardKinematics(model_, data_, q, v);
    pin::computeJointJacobians(model_, data_); // ?有什么用？
    pin::updateFramePlacements(model_, data_);
    pin::crba(model_, data_, q);
    data_.M.triangularView<Eigen::StrictlyLower>() = data_.M.transpose().triangularView<Eigen::StrictlyLower>();
    pin::nonLinearEffects(model_, data_, q, v);
    for (size_t i = 0; i < 4; ++i)
    {
        Eigen::Matrix<double, 6, Eigen::Dynamic> jac;
        jac.setZero(6, model_.nv);
        pin::getFrameJacobian(model_, data_, wbc_settings_.contact_ids[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);
        j_.middleRows(3 * i, 3) = jac.template topRows<3>();
    }

    // For not contact motion task
    pin::computeJointJacobiansTimeVariation(model_, data_, q, v);

    for (size_t i = 0; i < nc_; ++i)
    {
        Eigen::Matrix<double, 6, Eigen::Dynamic> jac;
        jac.setZero(6, model_.nv);
        pin::getFrameJacobianTimeVariation(model_, data_, wbc_settings_.contact_ids[i],
                                           pinocchio::LOCAL_WORLD_ALIGNED, jac);
        dj_.middleRows(3 * i, 3) = jac.template topRows<3>();
    }
    djdq_ = dj_ * v;
}

void WbcBase::updateFeference(const VectorXd &q_ref, const VectorXd &v_ref)
{
    pin::forwardKinematics(model_, data_ref_, q_ref, v_ref);
    pin::computeJointJacobians(model_, data_ref_, q_ref);
    pin::updateFramePlacements(model_, data_ref_);
}

Task WbcBase::formulateFloatingBaseEomTask()
{
    // a = [M, -J^T, -S^T]
    // b = -nle

    MatrixXd s(model_.nv - 6, model_.nv);
    s.middleCols(0, 6).setZero();
    s.middleCols(6, model_.nv - 6).setIdentity();

    MatrixXd a = MatrixXd(model_.nv, num_decision_vars_);
    a << data_.M, -j_.transpose(), -s.transpose();
    VectorXd b = -data_.nle;

    return {a, b, MatrixXd(), VectorXd()};
}

Task WbcBase::formulateTorqueLimitsTask()
{
    // d = [0, 0, I,
    //      0, 0,-I];
    // f = [tau_upper,
    //     -tau_lower]

    MatrixXd d(2 * (model_.nv - 6), num_decision_vars_);
    d.setZero();
    MatrixXd i = MatrixXd::Identity(model_.nv - 6, model_.nv - 6);
    d.block(0, model_.nv + 3 * nc_, model_.nv - 6, model_.nv - 6) = i;
    d.block(model_.nv - 6, model_.nv + 3 * nc_, model_.nv - 6, model_.nv - 6) = -i;

    VectorXd f(2 * (model_.nv - 6));
    f << model_.effortLimit.tail(model_.nv - 6), model_.effortLimit.tail(model_.nv - 6);

    return {MatrixXd(), VectorXd(), d, f};
}

Task WbcBase::formulateNoContactMotionTask()
{
    // a = [Jc, 0, 0,
    //      Jc, 0, 0];
    // b = -djdq

    MatrixXd a(3 * num_contacts_, num_decision_vars_);
    VectorXd b(a.rows());
    a.setZero();
    b.setZero();
    size_t j = 0;
    for (size_t i = 0; i < nc_; i++)
    {
        if (contact_state_[i])
        {
            a.block(3 * j, 0, 3, model_.nv) = j_.middleRows(3 * i, 3);
            b.segment(3 * j, 3) = -djdq_.segment(3 * i, 3);
            j++;
        }
    }

    return {a, b, MatrixXd(), VectorXd()};
}

Task WbcBase::formulateFrictionConeTask()
{
    // 摆动腿力为0约束
    MatrixXd a(3 * (nc_ - num_contacts_), num_decision_vars_);
    a.setZero();
    size_t j = 0;
    for (size_t i = 0; i < nc_; ++i)
    {
        if (!contact_state_[i])
        {
            a.block(3 * j++, model_.nv + 3 * i, 3, 3) = MatrixXd::Identity(3, 3);
        }
    }
    VectorXd b(a.rows());
    b.setZero();

    // 支撑腿摩擦锥约束
    MatrixXd frictionPyramic(5, 3);
    frictionPyramic << 0, 0, -1,  // fz <= 0
        1, 0, -wbc_settings_.mu,  // fx <= mu * fz
        -1, 0, -wbc_settings_.mu, // fx >= -mu * fz
        0, 1, -wbc_settings_.mu,  // fy <= mu * fz
        0, -1, -wbc_settings_.mu; // fy >= -mu * fz

    MatrixXd d(5 * num_contacts_ + 3 * (nc_ - num_contacts_), num_decision_vars_);
    d.setZero();
    j = 0;
    for (size_t i = 0; i < nc_; ++i)
    {
        if (contact_state_[i])
        {
            d.block(5 * j++, model_.nv + 3 * i, 5, 3) = frictionPyramic;
        }
    }
    VectorXd f = Eigen::VectorXd::Zero(d.rows());

    return {a, b, d, f};
}

Task WbcBase::formulateBaseAccelTask(const VectorXd &a_ref)
{
    MatrixXd a(6, num_decision_vars_);
    a.setZero();
    a.block(0, 0, 6, 6) = MatrixXd::Identity(6, 6);

    Vector6 b = a_ref.head(6);
    return {a, b, MatrixXd(), VectorXd()};
}

Task WbcBase::formulateSwingLegTask(const VectorXd &q_ref, const VectorXd &v_ref,
                                    const VectorXd &q, const VectorXd &v)
{
    // a = [Jsw, 0, 0,
    //      Jsw, 0, 0];
    // b = -djdq + a_feet

    MatrixXd a(3 * (nc_ - num_contacts_), num_decision_vars_);
    VectorXd b(3 * (nc_ - num_contacts_));
    a.setZero();
    b.setZero();
    size_t j = 0;
    Vector3d pos_foot_ref, pos_foot_measured, vel_foot_ref, vel_foot_measured;
    for (size_t i = 0; i < nc_; ++i)
    {
        if (!contact_state_[i])
        {
            pos_foot_ref = data_ref_.oMf[wbc_settings_.contact_ids[i]].translation();
            pos_foot_measured = data_.oMf[wbc_settings_.contact_ids[i]].translation();
            vel_foot_ref = pin::getFrameVelocity(model_, data_ref_, wbc_settings_.contact_ids[i], pinocchio::LOCAL_WORLD_ALIGNED).linear();
            vel_foot_measured = pin::getFrameVelocity(model_, data_, wbc_settings_.contact_ids[i], pinocchio::LOCAL_WORLD_ALIGNED).linear();

            Vector3d accel = wbc_settings_.kp_sw * (pos_foot_ref - pos_foot_measured) +
                             wbc_settings_.kd_sw * (vel_foot_ref - vel_foot_measured);
            a.block(3 * j, 0, 3, model_.nv) = j_.block(3 * i, 0, 3, model_.nv);
            b.segment(3 * j, 3) = accel - djdq_.segment(3 * i, 3);
            j++;
        }
    }

    return {a, b, MatrixXd(), VectorXd()};
}

Task WbcBase::formulateContactForceTask(const VectorXd &force_ref) const
{
    MatrixXd a(3 * nc_, num_decision_vars_);
    VectorXd b(3 * nc_);
    a.setZero();

    for (size_t i = 0; i < nc_; ++i)
    {
        a.block(3 * i, model_.nv + 3 * i, 3, 3) = MatrixXd::Identity(3, 3);
    }
    b = force_ref; // ? 支撑腿和摆动腿都包含吗？

    return {a, b, MatrixXd(), VectorXd()};
}