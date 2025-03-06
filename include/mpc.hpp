#pragma once

#include "kinodynamics_ocp.hpp"

struct MPCSettings
{
};

class MPC
{
public:
    MPC(const MPCSettings &settings, std::shared_ptr<KinodynamicsOCP> problem);


private:
    MPCSettings settings_;
    std::shared_ptr<KinodynamicsOCP> problem_;
};