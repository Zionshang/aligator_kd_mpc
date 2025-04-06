#include "kinodynamics.hpp"

#include <aligator/modelling/centroidal/centroidal-friction-cone.hpp>
#include <aligator/modelling/centroidal/centroidal-wrench-cone.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/dynamics/kinodynamics-fwd.hpp>
#include <aligator/modelling/multibody/center-of-mass-translation.hpp>
#include <aligator/modelling/multibody/centroidal-momentum-derivative.hpp>
#include <aligator/modelling/multibody/centroidal-momentum.hpp>
#include <aligator/modelling/multibody/frame-placement.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/modelling/multibody/frame-velocity.hpp>

namespace simple_mpc
{
  using namespace aligator;
  using MultibodyPhaseSpace = proxsuite::nlp::MultibodyPhaseSpace<double>;
  using KinodynamicsFwdDynamics = dynamics::KinodynamicsFwdDynamicsTpl<double>;
  using CentroidalWrenchConeResidual = CentroidalWrenchConeResidualTpl<double>;
  using CentroidalFrictionConeResidual = CentroidalFrictionConeResidualTpl<double>;
  using FramePlacementResidual = FramePlacementResidualTpl<double>;
  using FrameTranslationResidual = FrameTranslationResidualTpl<double>;
  using FrameVelocityResidual = FrameVelocityResidualTpl<double>;
  using CenterOfMassTranslationResidual = CenterOfMassTranslationResidualTpl<double>;
  using IntegratorSemiImplEuler = dynamics::IntegratorSemiImplEulerTpl<double>;

  KinodynamicsOCP::KinodynamicsOCP(const KinodynamicsSettings &settings, const RobotModelHandler &model_handler)
      : settings_(settings), model_handler_(model_handler), problem_(nullptr)
  {
    nq_ = model_handler.getModel().nq;
    nv_ = model_handler.getModel().nv;
    ndx_ = 2 * model_handler.getModel().nv;
    nu_ = nv_ - 6 + settings_.force_size * (int)model_handler_.getFeetNames().size();
    x0_ = model_handler_.getReferenceState();
    control_ref_.resize(nu_);
    control_ref_.setZero();
  }

  StageModel KinodynamicsOCP::createStage(const std::map<std::string, bool> &contact_phase,
                                          const std::map<std::string, pinocchio::SE3> &contact_pose,
                                          const std::map<std::string, Eigen::VectorXd> &contact_force)
  {
    auto space = MultibodyPhaseSpace(model_handler_.getModel());
    auto rcost = CostStack(space, nu_);
    std::vector<bool> contact_states;
    for (auto const &x : contact_phase)
    {
      contact_states.push_back(x.second);
    }

    computeControlFromForces(contact_force);

    rcost.addCost("state_cost", QuadraticStateCost(space, nu_, x0_, settings_.w_x)); // ? 期望状态后续有更新吗？
    rcost.addCost("control_cost", QuadraticControlCost(space, control_ref_, settings_.w_u));

    for (auto const &name : model_handler_.getFeetNames()) // todo: 改成只考虑摆动腿
    {
      FrameTranslationResidual frame_residual = FrameTranslationResidual(
          space.ndx(), nu_, model_handler_.getModel(), contact_pose.at(name).translation(),
          model_handler_.getFootId(name)); // ? mpc过程中在哪里更新了contact_pose的值？

      rcost.addCost(name + "_pose_cost", QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    }

    KinodynamicsFwdDynamics ode = KinodynamicsFwdDynamics(
        space, model_handler_.getModel(), settings_.gravity, contact_states, model_handler_.getFeetIds(),
        settings_.force_size);
    IntegratorSemiImplEuler dyn_model = IntegratorSemiImplEuler(ode, settings_.timestep);
    StageModel stm = StageModel(rcost, dyn_model);

    if (settings_.kinematics_limits) // todo: 测试会影响多少计算时间
    {
      StateErrorResidual state_fn = StateErrorResidual(space, nu_, space.neutral());
      std::vector<int> state_id;
      for (int i = 6; i < nv_; i++)
      {
        state_id.push_back(i);
      }
      FunctionSliceXpr state_slice = FunctionSliceXpr(state_fn, state_id);
      stm.addConstraint(state_slice, BoxConstraint(-settings_.qmax, -settings_.qmin));
    }

    Motion v_ref = Motion::Zero();
    int i = 0;
    for (auto const &name : model_handler_.getFeetNames())
    {
      if (contact_phase.at(name))
      {
        // 摩擦力约束
        if (settings_.force_cone)
        {
          CentroidalFrictionConeResidual friction_residual =
              CentroidalFrictionConeResidual(space.ndx(), nu_, i, settings_.mu, 1e-4);
          stm.addConstraint(friction_residual, NegativeOrthant());
        }

        // 支撑腿速度为 0 约束
        FrameVelocityResidual frame_vel = FrameVelocityResidual(
            space.ndx(), nu_, model_handler_.getModel(), v_ref, model_handler_.getFootId(name), pinocchio::LOCAL);
        std::vector<int> vel_id = {0, 1, 2}; // 只考虑平移速度
        FunctionSliceXpr vel_slice = FunctionSliceXpr(frame_vel, vel_id);
        stm.addConstraint(vel_slice, EqualityConstraint());
      }
      i++;
    }

    return stm;
  }

  void KinodynamicsOCP::setReferenceFootPose(const std::size_t t, const std::string &ee_name, const pinocchio::SE3 &pose_ref)
  {
    CostStack *cs = getCostStack(t);
    QuadraticResidualCost *qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
    FrameTranslationResidual *cfr = qrc->getResidual<FrameTranslationResidual>();
    cfr->setReference(pose_ref.translation());
  }

  // ? 这里是不是有问题，终端成本应该没有_pose_cost
  void KinodynamicsOCP::setTerminalReferencePose(const std::string &ee_name, const pinocchio::SE3 &pose_ref)
  {
    CostStack *cs = getTerminalCostStack();
    QuadraticResidualCost *qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
    FrameTranslationResidual *cfr = qrc->getResidual<FrameTranslationResidual>();
    cfr->setReference(pose_ref.translation());
  }

  void KinodynamicsOCP::computeControlFromForces(const std::map<std::string, Eigen::VectorXd> &force_refs)
  {
    for (std::size_t i = 0; i < model_handler_.getFeetNames().size(); i++)
    {
      if (settings_.force_size != force_refs.at(model_handler_.getFootName(i)).size())
      {
        throw std::runtime_error("force size in settings does not match reference force size");
      }
      control_ref_.segment((long)i * settings_.force_size, settings_.force_size) =
          force_refs.at(model_handler_.getFootName(i));
    }
  }

  const Eigen::VectorXd KinodynamicsOCP::getReferenceForce(const std::size_t i, const std::string &ee_name)
  {
    std::vector<std::string> hname = model_handler_.getFeetNames();
    std::vector<std::string>::iterator it = std::find(hname.begin(), hname.end(), ee_name);
    long id = it - hname.begin();

    return getReferenceControl(i).segment(id * settings_.force_size, settings_.force_size);
  }

  const Eigen::VectorXd KinodynamicsOCP::getVelocityBase(const std::size_t t)
  {
    CostStack *cs = getCostStack(t);
    QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
    return qc->getTarget().segment(nq_, 6);
  }

  void KinodynamicsOCP::setVelocityBase(const std::size_t t, const ConstVectorRef &velocity_base)
  {
    if (velocity_base.size() != 6)
    {
      throw std::runtime_error("velocity_base size should be 6");
    }
    CostStack *cs = getCostStack(t);
    QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
    x0_.segment(nq_, 6) = velocity_base;
    qc->setTarget(x0_);
  }

  const Eigen::VectorXd KinodynamicsOCP::getPoseBase(const std::size_t t)
  {
    CostStack *cs = getCostStack(t);
    QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
    return qc->getTarget().head(7);
  };

  void KinodynamicsOCP::setPoseBase(const std::size_t t, const ConstVectorRef &pose_base)
  {
    if (pose_base.size() != 7)
    {
      throw std::runtime_error("pose_base size should be 7");
    }
    CostStack *cs = getCostStack(t);
    QuadraticStateCost *qc = cs->getComponent<QuadraticStateCost>("state_cost");
    x0_.head(7) = pose_base;
    qc->setTarget(x0_);
  }

  const Eigen::VectorXd KinodynamicsOCP::getProblemState(const RobotDataHandler &data_handler)
  {
    return data_handler.getState();
  }

  size_t KinodynamicsOCP::getContactSupport(const std::size_t t)
  {
    KinodynamicsFwdDynamics *ode =
        problem_->stages_[t]->getDynamics<IntegratorSemiImplEuler>()->getDynamics<KinodynamicsFwdDynamics>();

    size_t active_contacts = 0;
    for (auto const contact : ode->contact_states_)
    {
      if (contact)
      {
        active_contacts += 1;
      }
    }
    return active_contacts;
  }

  std::vector<bool> KinodynamicsOCP::getContactState(const std::size_t t)
  {
    KinodynamicsFwdDynamics *ode =
        problem_->stages_[t]->getDynamics<IntegratorSemiImplEuler>()->getDynamics<KinodynamicsFwdDynamics>();
    assert(ode != nullptr);
    return ode->contact_states_;
  }

  CostStack KinodynamicsOCP::createTerminalCost()
  {
    auto ter_space = MultibodyPhaseSpace(model_handler_.getModel());
    auto term_cost = CostStack(ter_space, nu_);

    term_cost.addCost("state_cost", QuadraticStateCost(ter_space, nu_, x0_, settings_.w_x));

    return term_cost;
  }

  // todo: 传参改为传递结构体
  void KinodynamicsOCP::createProblem(const ConstVectorRef &x0,
                                      const size_t horizon,
                                      const int force_size,
                                      const double gravity) // todo: double 改为 Eigen::Vector3d
  {
    std::vector<std::map<std::string, bool>> contact_phases;
    std::vector<std::map<std::string, pinocchio::SE3>> contact_poses;
    std::vector<std::map<std::string, Eigen::VectorXd>> contact_forces;

    Eigen::VectorXd force_ref(force_size);
    force_ref.setZero();
    force_ref[2] = -model_handler_.getMass() * gravity / (double)model_handler_.getFeetNames().size();

    std::map<std::string, bool> contact_phase;
    std::map<std::string, pinocchio::SE3> contact_pose;
    std::map<std::string, Eigen::VectorXd> contact_force;
    for (auto &name : model_handler_.getFeetNames())
    {
      contact_phase.insert({name, true});
      contact_pose.insert({name, pinocchio::SE3::Identity()}); // ? 这里设置为Identity是否合理？
      contact_force.insert({name, force_ref});
    }

    for (size_t i = 0; i < horizon; i++)
    {
      contact_phases.push_back(contact_phase);
      contact_poses.push_back(contact_pose);
      contact_forces.push_back(contact_force);
    }

    std::vector<xyz::polymorphic<StageModel>> stage_models;
    for (std::size_t i = 0; i < horizon; i++)
    {
      StageModel stage = createStage(contact_phases[i], contact_poses[i], contact_forces[i]);
      stage_models.push_back(std::move(stage));
    }

    problem_ = std::make_unique<TrajOptProblem>(x0, std::move(stage_models), createTerminalCost());
  }

  void KinodynamicsOCP::setReferenceControl(const std::size_t t, const ConstVectorRef &u_ref)
  {
    CostStack *cs = getCostStack(t);
    QuadraticControlCost *qc = cs->getComponent<QuadraticControlCost>("control_cost");
    qc->setTarget(u_ref);
  }

  ConstVectorRef KinodynamicsOCP::getReferenceControl(const std::size_t t)
  {
    CostStack *cs = getCostStack(t);
    QuadraticControlCost *qc = cs->getComponent<QuadraticControlCost>("control_cost");
    return qc->getTarget();
  }

  CostStack *KinodynamicsOCP::getCostStack(std::size_t t)
  {
    if (t >= getSize())
    {
      throw std::runtime_error("Stage index exceeds stage vector size");
    }
    CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[t]->cost_);

    return cs;
  }

  CostStack *KinodynamicsOCP::getTerminalCostStack()
  {
    CostStack *cs = dynamic_cast<CostStack *>(&*problem_->term_cost_);

    return cs;
  }

} // namespace simple_mpc