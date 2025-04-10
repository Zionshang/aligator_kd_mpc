#include "planner/foot_planner.hpp"

FootPlanner::FootPlanner()
{
    height_swing = 0.2;
    offset_foot_ = 0.026;
    first_run_ = true;

    Matrix34d pos_shoulder; // position of shoulder
    pos_shoulder << 0.32, 0.32, -0.32, -0.32,
        0.18, -0.18, 0.18, -0.18,
        0.0, 0.0, 0.0, 0.0;
    for (int i = 0; i < 4; i++)
    {
        feet_radius_(i) = sqrt(pow(pos_shoulder(0, i), 2) + pow(pos_shoulder(1, i), 2));
        feet_init_angle(i) = atan2(pos_shoulder(1, i), pos_shoulder(0, i));
    }
}

void FootPlanner::update(const GaitState &gait_state, const BodyState &body_state, const FootState &feet_state,
                         FootState &feet_state_ref)
{
    const auto &pos_feet = feet_state.pos;
    auto &pos_feet_ref = feet_state_ref.pos;
    auto &vel_feet_ref = feet_state_ref.vel;

    phase_ = gait_state.phase;
    period_swing_ = gait_state.period_swing;
    period_stance_ = gait_state.period_stance;

    if (first_run_)
    {
        pos_start_ = pos_feet;
        first_run_ = false;
    }
    for (int i = 0; i < 4; ++i)
    {
        if (gait_state.contact(i) == 1)
        {
            if (phase_(i) < 0.5)
                pos_start_.col(i) = pos_feet.col(i);
            pos_feet_ref.col(i) = pos_start_.col(i);
            vel_feet_ref.col(i).setZero();
        }
        else
        {
            pos_end_.col(i) = calcFootholdPosition(body_state, i);
            pos_feet_ref.col(i) = calcReferenceFootPosition(i);
            vel_feet_ref.col(i) = calcReferenceFootVelocity(i);
        }
    }
}

Vector3d FootPlanner::calcFootholdPosition(const BodyState &body_state, int leg_id)
{
    const Vector3d pos_body = body_state.pos;
    const Vector3d vel_body = body_state.vel;
    const Vector3d angvel_body = body_state.angvel;

    // TODO: 是否需要改成相对于body系下
    // Translation in x,y axis
    next_step_(0) = vel_body(0) * (1 - phase_(leg_id)) * period_swing_ + vel_body(0) * period_stance_ / 2;
    next_step_(1) = vel_body(1) * (1 - phase_(leg_id)) * period_swing_ + vel_body(1) * period_stance_ / 2;
    next_step_(2) = 0;

    // rotation about z axis
    // todo: 修改这一项
    yaw_ = 0;
    next_yaw_ = angvel_body(2) * (1 - phase_(leg_id)) * period_swing_ + angvel_body(2) * period_stance_ / 2;
    next_step_(0) += feet_radius_(leg_id) * cos(yaw_ + feet_init_angle(leg_id) + next_yaw_);
    next_step_(1) += feet_radius_(leg_id) * sin(yaw_ + feet_init_angle(leg_id) + next_yaw_);

    Vector3d footholdPos;
    footholdPos = pos_body + next_step_;
    footholdPos(2) = offset_foot_;
    return footholdPos;
}

Vector3d FootPlanner::calcReferenceFootPosition(int leg_id)
{
    Vector3d footPos;

    footPos(0) = cycloidXYPosition(pos_start_.col(leg_id)(0), pos_end_.col(leg_id)(0), phase_(leg_id));
    footPos(1) = cycloidXYPosition(pos_start_.col(leg_id)(1), pos_end_.col(leg_id)(1), phase_(leg_id));
    footPos(2) = cycloidZPosition(pos_start_.col(leg_id)(2), height_swing + offset_foot_, phase_(leg_id));

    return footPos;
}

Vector3d FootPlanner::calcReferenceFootVelocity(int leg_id)
{
    Vector3d footVel;

    footVel(0) = cycloidXYVelocity(pos_start_.col(leg_id)(0), pos_end_.col(leg_id)(0), phase_(leg_id));
    footVel(1) = cycloidXYVelocity(pos_start_.col(leg_id)(1), pos_end_.col(leg_id)(1), phase_(leg_id));
    footVel(2) = cycloidZVelocity(height_swing + offset_foot_, phase_(leg_id));

    return footVel;
}

double FootPlanner::cycloidXYPosition(double start, double end, double phase)
{
    double phasePI = 2 * M_PI * phase;
    return (end - start) * (phasePI - sin(phasePI)) / (2 * M_PI) + start;
}

double FootPlanner::cycloidXYVelocity(double start, double end, double phase)
{
    double phasePI = 2 * M_PI * phase;
    return (end - start) * (1 - cos(phasePI)) / period_swing_;
}

double FootPlanner::cycloidZPosition(double start, double h, double phase)
{
    double phasePI = 2 * M_PI * phase;
    return h * (1 - cos(phasePI)) / 2 + start;
}

double FootPlanner::cycloidZVelocity(double h, double phase)
{
    double phasePI = 2 * M_PI * phase;
    return h * M_PI * sin(phasePI) / period_swing_;
}