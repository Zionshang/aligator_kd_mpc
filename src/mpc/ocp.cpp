#include "mpc/ocp.hpp"

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

  OCP::OCP(const OcpSettings &settings, const RobotModelHandler &model_handler)
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

  StageModel OCP::createStage(const std::map<std::string, bool> &contact_phase,
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

    // 机身位置成本
    FrameTranslationResidual body_trans_residual = FrameTranslationResidual(
        space.ndx(), nu_, model_handler_.getModel(), x0_.head(3), model_handler_.getBaseFrameId());
    rcost.addCost("body_trans_cost", QuadraticResidualCost(space, body_trans_residual, settings_.w_body_trans));

    // 机身旋转成本
    FramePlacementResidual body_place_residual = FramePlacementResidual(
        space.ndx(), nu_, model_handler_.getModel(), pinocchio::SE3::Identity(), model_handler_.getBaseFrameId());
    std::vector<int> body_rot_id{3, 4, 5};
    FunctionSliceXpr body_rot_residual = FunctionSliceXpr(body_place_residual, body_rot_id);
    rcost.addCost("body_rot_cost", QuadraticResidualCost(space, body_rot_residual, settings_.w_body_rot));

    // 机身速度成本
    FrameVelocityResidual body_vel_residual = FrameVelocityResidual(
        space.ndx(), nu_, model_handler_.getModel(), Motion::Zero(), model_handler_.getBaseFrameId(), pinocchio::LOCAL_WORLD_ALIGNED);
    rcost.addCost("body_vel_cost", QuadraticResidualCost(space, body_vel_residual, settings_.w_body_vel));

    // 腿的关节位置和速度成本
    StateErrorResidual state_residual = StateErrorResidual(space, nu_, space.neutral());
    std::vector<int> leg_pos_id, leg_vel_id;
    for (int i = 6; i < nv_; i++)
      leg_pos_id.push_back(i);
    for (int i = nv_ + 6; i < nv_ + nv_; i++)
      leg_vel_id.push_back(i);
    FunctionSliceXpr leg_pos_residual = FunctionSliceXpr(state_residual, leg_pos_id);
    FunctionSliceXpr leg_vel_residual = FunctionSliceXpr(state_residual, leg_vel_id);
    rcost.addCost("leg_pos_cost", QuadraticResidualCost(space, leg_pos_residual, settings_.w_leg_pos));
    rcost.addCost("leg_vel_cost", QuadraticResidualCost(space, leg_pos_residual, settings_.w_leg_vel));

    // 控制输入成本
    rcost.addCost("control_cost", QuadraticControlCost(space, control_ref_, settings_.w_u));

    for (auto const &name : model_handler_.getFeetNames())
    {
      FrameTranslationResidual frame_residual = FrameTranslationResidual(
          space.ndx(), nu_, model_handler_.getModel(), contact_pose.at(name).translation(),
          model_handler_.getFootId(name));

      rcost.addCost(name + "_pose_cost", QuadraticResidualCost(space, frame_residual, settings_.w_frame));
    }

    KinodynamicsFwdDynamics ode = KinodynamicsFwdDynamics(
        space, model_handler_.getModel(), settings_.gravity, contact_states, model_handler_.getFeetIds(),
        settings_.force_size);
    IntegratorSemiImplEuler dyn_model = IntegratorSemiImplEuler(ode, settings_.timestep);
    StageModel stm = StageModel(rcost, dyn_model);

    if (settings_.kinematics_limits)
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
        CentroidalFrictionConeResidual friction_residual =
            CentroidalFrictionConeResidual(space.ndx(), nu_, i, settings_.mu, 1e-4);
        stm.addConstraint(friction_residual, NegativeOrthant());

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

  void OCP::setReferenceFootPose(const std::size_t t, const std::string &ee_name, const pinocchio::SE3 &pose_ref)
  {
    CostStack *cs = getCostStack(t);
    QuadraticResidualCost *qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
    FrameTranslationResidual *cfr = qrc->getResidual<FrameTranslationResidual>();
    cfr->setReference(pose_ref.translation());
  }

  void OCP::setReferenceState(const std::size_t t, const ConstVectorRef &x_ref)
  {
    CostStack *cs = getCostStack(t);
    QuadraticResidualCost *body_trans_cost = cs->getComponent<QuadraticResidualCost>("body_trans_cost");
    FrameTranslationResidual *body_trans_residual = body_trans_cost->getResidual<FrameTranslationResidual>();

    QuadraticResidualCost *body_rot_cost = cs->getComponent<QuadraticResidualCost>("body_rot_cost");
    FramePlacementResidual *body_rot_residual = body_rot_cost->getResidual<FramePlacementResidual>();

    QuadraticResidualCost *body_vel_cost = cs->getComponent<QuadraticResidualCost>("body_vel_cost");
    FrameVelocityResidual *body_vel_residual = body_vel_cost->getResidual<FrameVelocityResidual>();

    // QuadraticResidualCost *leg_pos_cost = cs->getComponent<QuadraticResidualCost>("leg_pos_cost");
    // StateErrorResidual *leg_pos_residual = leg_pos_cost->getResidual<StateErrorResidual>();

    // QuadraticResidualCost *leg_vel_cost = cs->getComponent<QuadraticResidualCost>("leg_vel_cost");
    // StateErrorResidual *leg_vel_residual = leg_pos_cost->getResidual<StateErrorResidual>();

    body_trans_residual->setReference(x_ref.head(3));
    Eigen::Quaterniond quat(x_ref(6), x_ref(3), x_ref(4), x_ref(5));
    pinocchio::SE3 se3 = pinocchio::SE3::Identity();
    se3.rotation(quat.toRotationMatrix());
    body_rot_residual->setReference(se3);
    std::cout << "Test" << std::endl;

    pinocchio::Motion v_ref = pinocchio::Motion(x_ref.segment(nq_, 6));
    body_vel_residual->setReference(v_ref);
    // qsc->setTarget(x_ref);
    std::cout << "Test" << std::endl;

  }

  void OCP::setTerminalReferenceState(const ConstVectorRef &x_ref)
  {
    CostStack *cs = getTerminalCostStack();
    QuadraticStateCost *qsc = cs->getComponent<QuadraticStateCost>("term_state_cost");
    qsc->setTarget(x_ref);
  }

  void OCP::computeControlFromForces(const std::map<std::string, Eigen::VectorXd> &force_refs)
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

  const Eigen::VectorXd OCP::getReferenceForce(const std::size_t i, const std::string &ee_name)
  {
    std::vector<std::string> hname = model_handler_.getFeetNames();
    std::vector<std::string>::iterator it = std::find(hname.begin(), hname.end(), ee_name);
    long id = it - hname.begin();

    return getReferenceControl(i).segment(id * settings_.force_size, settings_.force_size);
  }

  const Eigen::VectorXd OCP::getProblemState(const RobotDataHandler &data_handler)
  {
    return data_handler.getState();
  }

  size_t OCP::getContactSupport(const std::size_t t)
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

  std::vector<bool> OCP::getContactState(const std::size_t t)
  {
    KinodynamicsFwdDynamics *ode =
        problem_->stages_[t]->getDynamics<IntegratorSemiImplEuler>()->getDynamics<KinodynamicsFwdDynamics>();
    assert(ode != nullptr);
    return ode->contact_states_;
  }

  CostStack OCP::createTerminalCost()
  {
    auto ter_space = MultibodyPhaseSpace(model_handler_.getModel());
    auto term_cost = CostStack(ter_space, nu_);

    term_cost.addCost("term_state_cost", QuadraticStateCost(ter_space, nu_, x0_, settings_.w_x));

    return term_cost;
  }

  // todo: 传参改为传递结构体
  void OCP::createProblem(const ConstVectorRef &x0,
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

    // todo: 更优雅的方法
    const auto &model = model_handler_.getModel();
    pin::Data data(model);
    pin::forwardKinematics(model, data, x0.head(model.nq));
    pin::updateFramePlacements(model, data);

    for (auto &name : model_handler_.getFeetNames())
    {
      contact_phase.insert({name, true});
      contact_pose.insert({name, data.oMf[model_handler_.getFootId(name)]});
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

  void OCP::setReferenceControl(const std::size_t t, const ConstVectorRef &u_ref)
  {
    CostStack *cs = getCostStack(t);
    QuadraticControlCost *qc = cs->getComponent<QuadraticControlCost>("control_cost");
    qc->setTarget(u_ref);
  }

  ConstVectorRef OCP::getReferenceControl(const std::size_t t)
  {
    CostStack *cs = getCostStack(t);
    QuadraticControlCost *qc = cs->getComponent<QuadraticControlCost>("control_cost");
    return qc->getTarget();
  }

  CostStack *OCP::getCostStack(std::size_t t)
  {
    if (t >= getSize())
    {
      throw std::runtime_error("Stage index exceeds stage vector size");
    }
    CostStack *cs = dynamic_cast<CostStack *>(&*problem_->stages_[t]->cost_);

    return cs;
  }

  CostStack *OCP::getTerminalCostStack()
  {
    CostStack *cs = dynamic_cast<CostStack *>(&*problem_->term_cost_);

    return cs;
  }
  const pinocchio::SE3 OCP::getReferenceFootPose(const std::size_t t, const std::string &ee_name)
  {
    CostStack *cs = getCostStack(t);
    QuadraticResidualCost *qrc = cs->getComponent<QuadraticResidualCost>(ee_name + "_pose_cost");
    FrameTranslationResidual *cfr = qrc->getResidual<FrameTranslationResidual>();
    SE3 ref = SE3::Identity();
    ref.translation() = cfr->getReference();
    return ref;
  }

} // namespace simple_mpc