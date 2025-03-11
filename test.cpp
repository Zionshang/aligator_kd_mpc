#include "fwd.hpp"
#include "kinodynamics.hpp"
#include "mpc.hpp"
#include "robot-handler.hpp"
#include "test_utils.hpp"

int main(int argc, char const *argv[])
{
    RobotModelHandler model_handler = getTalosModelHandler();
    RobotDataHandler data_handler(model_handler);

    KinodynamicsSettings settings = getKinodynamicsSettings(model_handler);
    auto problem = std::make_shared<KinodynamicsOCP>(settings, model_handler);
    KinodynamicsOCP &kinoproblem = *problem;
    const std::size_t T = 100;
    const double support_force = -model_handler.getMass() * settings.gravity[2];
    Eigen::VectorXd f1(6);
    f1 << 0, 0, support_force, 0, 0, 0;

    kinoproblem.createProblem(model_handler.getReferenceState(), T, 6, -settings.gravity[2], true);

    MPCSettings mpc_settings;
    mpc_settings.max_iters = 1;

    mpc_settings.support_force = support_force;

    mpc_settings.TOL = 1e-6;
    mpc_settings.mu_init = 1e-8;
    mpc_settings.num_threads = 8;

    mpc_settings.swing_apex = 0.1;
    mpc_settings.T_fly = 80;
    mpc_settings.T_contact = 20;
    mpc_settings.timestep = 0.01;

    MPC mpc = MPC(mpc_settings, problem);

    std::vector<std::map<std::string, bool>> contact_states;
    // std::vector<std::vector<bool>> contact_states;
    for (std::size_t i = 0; i < 10; i++)
    {
        std::map<std::string, bool> contact_state;
        contact_state.insert({model_handler.getFootName(0), true});
        contact_state.insert({model_handler.getFootName(1), true});
        contact_states.push_back(contact_state);
    }
    for (std::size_t i = 0; i < 50; i++)
    {
        std::map<std::string, bool> contact_state;
        contact_state.insert({model_handler.getFootName(0), true});
        contact_state.insert({model_handler.getFootName(1), false});
        contact_states.push_back(contact_state);
    }
    for (std::size_t i = 0; i < 10; i++)
    {
        std::map<std::string, bool> contact_state;
        contact_state.insert({model_handler.getFootName(0), true});
        contact_state.insert({model_handler.getFootName(1), true});
        contact_states.push_back(contact_state);
    }
    for (std::size_t i = 0; i < 50; i++)
    {
        std::map<std::string, bool> contact_state;
        contact_state.insert({model_handler.getFootName(0), false});
        contact_state.insert({model_handler.getFootName(1), true});
        contact_states.push_back(contact_state);
    }

    mpc.generateCycleHorizon(contact_states);

    for (std::size_t i = 0; i < 10; i++)
    {
        mpc.iterate(model_handler.getReferenceState());
    }

    Eigen::VectorXd xdot = mpc.getStateDerivative(0);
    return 0;
}
