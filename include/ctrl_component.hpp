#pragma once
#include "fwd.hpp"
struct BodyState
{
    Vector3d pos;            // position of body, expressed in world frame
    Eigen::Quaterniond quat; // quaternion of body relative to world frame
    Vector3d vel;            // velocity of body, expressed in world frame
    Vector3d angvel;         // angluar velocity of body, expressed in world frame

    BodyState()
    {
        pos.setZero();
        quat.setIdentity();
        vel.setZero();
        angvel.setZero();
    }
};

struct FootState
{
    Matrix34d pos;          // position of foot, expressed in world frame
    Matrix34d vel;          // velocity of foot, expressed in world frame

    FootState()
    {
        pos.setZero();
        vel.setZero();
    }
};