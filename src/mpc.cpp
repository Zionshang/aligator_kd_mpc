#include "mpc.hpp"

MPC::MPC(const MPCSettings &settings, std::shared_ptr<KinodynamicsOCP> problem)
    : settings_(settings), problem_(problem)
{
}