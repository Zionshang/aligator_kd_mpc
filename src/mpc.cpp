///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "mpc.hpp"
#include "utils/logger.hpp"
#include <aligator/modelling/multibody/frame-translation.hpp>

namespace simple_mpc
{
  using namespace aligator;
  using FrameTranslationResidual = FrameTranslationResidualTpl<double>;

  constexpr std::size_t maxiters = 100;

  MPC::MPC(const MPCSettings &settings, std::shared_ptr<KinodynamicsOCP> problem)
      : settings_(settings), ocp_handler_(problem)
  {
    data_handler_ = std::make_shared<RobotDataHandler>(ocp_handler_->getModelHandler());
    data_handler_->updateInternalData(ocp_handler_->getModelHandler().getReferenceState(), true);
    std::map<std::string, Eigen::Vector3d> starting_poses;
    for (auto const &name : ocp_handler_->getModelHandler().getFeetNames())
    {
      starting_poses.insert({name, data_handler_->getFootPose(name).translation()});

      relative_feet_poses_.insert(
          {name, data_handler_->getBaseFramePose().inverse() * data_handler_->getFootPose(name)});
    }
    foot_trajectories_ = FootTrajectory(
        starting_poses, settings_.swing_apex, settings_.T_fly, settings_.T_contact, ocp_handler_->getSize());

    foot_trajectories_.updateApex(settings.swing_apex);
    x0_ = ocp_handler_->getProblemState(*data_handler_);
    x_ref_.assign(ocp_handler_->getSize(), x0_);

    solver_ = std::make_unique<SolverProxDDP>(settings_.TOL, settings_.mu_init, maxiters, aligator::VerboseLevel::QUIET);
    solver_->rollout_type_ = aligator::RolloutType::LINEAR;

    if (settings_.num_threads > 1)
    {
      solver_->linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
      solver_->setNumThreads(settings_.num_threads);
    }
    else
      solver_->linear_solver_choice = aligator::LQSolverChoice::SERIAL;
    solver_->force_initial_condition_ = true;
    // solver_->reg_min = 1e-6;

    ee_names_ = ocp_handler_->getModelHandler().getFeetNames();
    Eigen::VectorXd force_ref(ocp_handler_->getReferenceForce(0, ocp_handler_->getModelHandler().getFootName(0)));

    std::map<std::string, bool> contact_states;
    std::map<std::string, pinocchio::SE3> contact_poses;
    std::map<std::string, Eigen::VectorXd> force_map;

    for (auto const &name : ee_names_)
    {
      contact_states.insert({name, true});
      contact_poses.insert({name, data_handler_->getFootPose(name)});
      force_map.insert({name, force_ref});
    }

    for (std::size_t i = 0; i < ocp_handler_->getSize(); i++)
    {
      xs_.push_back(x0_);
      us_.push_back(ocp_handler_->getReferenceControl(0));

      std::shared_ptr<StageModel> sm = std::make_shared<StageModel>(ocp_handler_->createStage(contact_states, contact_poses, force_map));
      standing_horizon_.push_back(sm);
      standing_horizon_data_.push_back(sm->createData());
    }
    xs_.push_back(x0_);

    solver_->setup(ocp_handler_->getProblem());
    solver_->run(ocp_handler_->getProblem(), xs_, us_);

    xs_ = solver_->results_.xs;
    us_ = solver_->results_.us;
    Ks_ = solver_->results_.getCtrlFeedbacks();

    as_.resize(ocp_handler_->getSize());
    // saveVectorsToCsv("initial_xs.csv", xs_);
    // std::cout << ocp_handler_->getReferencePose(0, "FL_foot_link");
    // std::cout << ocp_handler_->getReferencePose(ocp_handler_->getSize()-1, "FL_foot_link");

    solver_->max_iters = settings_.max_iters;

    now_ = WALKING;
    pose_base_ = x0_.head<7>();
    velocity_base_.setZero();
    next_pose_.setZero();
    twist_vect_.setZero();
  }

  void MPC::generateCycleHorizon(const std::vector<std::map<std::string, bool>> &contact_states)
  {
    contact_states_ = contact_states;

    // Guarantee that cycle horizon size is higher than problem size
    int m = int(ocp_handler_->getSize()) / int(contact_states.size());
    for (int i = 0; i < m; i++)
    {
      std::vector<std::map<std::string, bool>> copy_vec = contact_states;
      contact_states_.insert(contact_states_.end(), copy_vec.begin(), copy_vec.end());
    }

    // Generate contact switch timings 落地时刻记录
    for (auto const &name : ee_names_)
    {
      foot_land_times_.insert({name, std::vector<int>()});
      for (size_t i = 1; i < contact_states_.size(); i++)
      {
        // 从摆动腿变为支撑腿
        if (contact_states_[i].at(name) and !contact_states_[i - 1].at(name))
        {
          // !这里要加上ocp_handler_->getSize()，是因为mpc首先会求一次完整周期的stance步态，
          // !利用这个stance步态进行热启动，swing步态处于第二个周期，所以要加上一个周期的时间
          foot_land_times_.at(name).push_back((int)(i + ocp_handler_->getSize()));
        }
      }
      // 保证首尾连续性，如果最后时刻是摆动腿且第一个时刻是支撑腿
      if (!contact_states_.back().at(name) and contact_states_[0].at(name))
        foot_land_times_.at(name).push_back((int)(contact_states_.size() - 1 + ocp_handler_->getSize()));
    }

    // Generate the model stages for cycle horizon
    for (auto const &state : contact_states_)
    {
      // 首先计算当前接触的末端个数
      int active_contacts = 0;
      for (auto const &contact : state)
      {
        if (contact.second)
          active_contacts += 1;
      }

      Eigen::VectorXd force_ref(ocp_handler_->getReferenceForce(0, ocp_handler_->getModelHandler().getFootName(0)));
      Eigen::VectorXd force_zero(ocp_handler_->getReferenceForce(0, ocp_handler_->getModelHandler().getFootName(0)));
      force_ref.setZero();
      force_zero.setZero();
      force_ref[2] = settings_.support_force / active_contacts;

      std::map<std::string, pinocchio::SE3> contact_poses;
      std::map<std::string, Eigen::VectorXd> force_map;

      for (auto const &name : ee_names_)
      {
        contact_poses.insert({name, data_handler_->getFootPose(name)});
        if (state.at(name))
          force_map.insert({name, force_ref});
        else
          force_map.insert({name, force_zero});
      }

      std::shared_ptr<StageModel> sm = std::make_shared<StageModel>(ocp_handler_->createStage(state, contact_poses, force_map));
      cycle_horizon_.push_back(sm);
      cycle_horizon_data_.push_back(sm->createData());
    }
  }

  void MPC::iterate(const ConstVectorRef &x, double current_time)
  {

    data_handler_->updateInternalData(x, false);

    // Recede all horizons
    recedeWithCycle(current_time);

    // Update the feet and CoM references
    updateStepTrackerReferences();

    // Recede previous solutions
    x0_ << ocp_handler_->getProblemState(*data_handler_);
    xs_.erase(xs_.begin());
    xs_[0] = x0_;
    xs_.push_back(xs_.back());

    us_.erase(us_.begin());
    us_.push_back(us_.back());

    ocp_handler_->getProblem().setInitState(x0_);

    // Run solver
    solver_->run(ocp_handler_->getProblem(), xs_, us_);

    // Collect results
    xs_ = solver_->results_.xs;
    us_ = solver_->results_.us;
    Ks_ = solver_->results_.getCtrlFeedbacks();
    for (int i = 0; i < ocp_handler_->getSize(); i++) // ? 放在MPC内部好还是外部好？
    {
      as_[i] = getStateDerivative(i).tail(ocp_handler_->getModelHandler().getModel().nv);
    }
  }

  void MPC::recedeWithCycle(double current_time)
  {
    if (now_ == WALKING or ocp_handler_->getContactSupport(ocp_handler_->getSize() - 1) < ee_names_.size())
    {
      constexpr double epsilon = 1e-6;
      // std::cout << "current_time: " << current_time << " last_recede_time_" << last_recede_time_ << " error: " << current_time - last_recede_time_ << std::endl;
      if (current_time - last_recede_time_ >= settings_.timestep - epsilon)
      {
        ocp_handler_->getProblem().replaceStageCircular(*cycle_horizon_[0]);
        solver_->cycleProblem(ocp_handler_->getProblem(), cycle_horizon_data_[0]); // ? 每次都是必须的吗？

        std::cout << "!!recedeWithCycle!!" << std::endl;
        rotate_vec_left(cycle_horizon_);
        rotate_vec_left(cycle_horizon_data_);
        rotate_vec_left(contact_states_);
        for (auto const &name : ee_names_)
        {
          if (contact_states_[contact_states_.size() - 1].at(name) and
              !contact_states_[contact_states_.size() - 2].at(name))
            foot_land_times_.at(name).push_back((int)(contact_states_.size() + ocp_handler_->getSize()));
        }
        updateCycleTiming(false); // ?为什么这里是false
        last_recede_time_ = current_time;
      }
    }
    else
    {
      ocp_handler_->getProblem().replaceStageCircular(*standing_horizon_[0]);
      solver_->cycleProblem(ocp_handler_->getProblem(), standing_horizon_data_[0]);

      rotate_vec_left(standing_horizon_);
      rotate_vec_left(standing_horizon_data_);

      updateCycleTiming(true); // ?为什么这里是true
    }
    // // Print out contact_states_ for debugging
    // std::vector<std::string> foot_names = {"FL_foot_link", "FR_foot_link", "HL_foot_link", "HR_foot_link"};
    // for (const auto &foot : foot_names)
    // {
    //   std::cout << foot.substr(0, 2) << " states: ";
    //   for (const auto &state : contact_states_)
    //     std::cout << state.at(foot) << " ";
    //   std::cout << std::endl;
    // }
  }

  // updateOnlyHorizon: 只更新mpc预测周期内的时间，不更新整个stage_models周期
  void MPC::updateCycleTiming(const bool updateOnlyHorizon)
  {
    for (auto const &name : ee_names_)
    {
      // 随着时间窗口的推进而向前“平移”
      for (size_t i = 0; i < foot_land_times_.at(name).size(); i++)
      {
        // 如果更新的是整个stage_models周期，或者foot_land_times 小于mpc预测周期，那么就减1
        // 站立阶段时，updateOnlyHorizon为true，窗口不需要更新
        if (!updateOnlyHorizon or foot_land_times_.at(name)[i] < (int)ocp_handler_->getSize())
          foot_land_times_.at(name)[i] -= 1;
      }
      // 如果第一个时间小于0，那么就删除
      if (!foot_land_times_.at(name).empty() and foot_land_times_.at(name)[0] < 0)
        foot_land_times_.at(name).erase(foot_land_times_.at(name).begin());
    }
    // std::cout << "foot_land_times_:" << std::endl;
    // for (const auto &pair : foot_land_times_)
    // {
    //   std::cout << pair.first << ": ";
    //   for (const auto &time : pair.second)
    //   {
    //     std::cout << time << " ";
    //   }
    //   std::cout << std::endl;
    // }
  }

  void MPC::updateStepTrackerReferences()
  {
    // Set reference state
    for (int i = 0; i < ocp_handler_->getSize(); i++)
    {
      ocp_handler_->setReferenceState(i, x_ref_[i]);
    }
    ocp_handler_->setTerminalReferenceState(x_ref_[ocp_handler_->getSize() - 1]);

    velocity_base_ = x_ref_[0].segment(getModelHandler().getModel().nq, 6);
    for (auto const &name : ee_names_)
    {
      int foot_land_time = -1;
      if (!foot_land_times_.at(name).empty())
        foot_land_time = foot_land_times_.at(name)[0];
      // std::cout << "name: " << name << " foot_land_time: " << foot_land_time << std::endl;
      bool update = true;
      // 如果足端即将落地，则不更新
      // ? 何时update?
      if (foot_land_time < settings_.T_fly)
        update = false;

      // Use the Raibert heuristics to compute the next foot pose
      twist_vect_[0] = -(data_handler_->getRefFootPose(name).translation()[1] - data_handler_->getBaseFramePose().translation()[1]);
      twist_vect_[1] = data_handler_->getRefFootPose(name).translation()[0] - data_handler_->getBaseFramePose().translation()[0];
      next_pose_.head(2) = data_handler_->getRefFootPose(name).translation().head(2);
      next_pose_.head(2) += (velocity_base_.head(2) + velocity_base_[5] * twist_vect_) * (settings_.T_fly + settings_.T_contact) * settings_.timestep;
      next_pose_[2] = data_handler_->getFootPose(name).translation()[2];

      // if (name == ee_names_[0])
      // {
      //   std::cout << "next_pose_: " << next_pose_.transpose() << std::endl;
      //   std::cout << "current_pose_: " << data_handler_->getFootPose(name).translation().transpose() << std::endl;
      // }

      foot_trajectories_.updateTrajectory(update, foot_land_time, data_handler_->getFootPose(name).translation(), next_pose_, name);
      pinocchio::SE3 pose_ref = pinocchio::SE3::Identity();
      for (unsigned long time = 0; time < ocp_handler_->getSize(); time++)
      {
        pose_ref.translation() = foot_trajectories_.getReference(name)[time];
        ocp_handler_->setReferenceFootPose(time, name, pose_ref);
      }
    }
  }

  TrajOptProblem &MPC::getTrajOptProblem()
  {
    return ocp_handler_->getProblem();
  }

  void MPC::switchToWalk(const Vector6d &velocity_base)
  {
    now_ = WALKING;
    velocity_base_ = velocity_base;
  }

  void MPC::switchToStand()
  {
    now_ = STANDING;
    velocity_base_.setZero();
  }

  void MPC::testCost()
  {
    double state_cost = 0, control_cost = 0, foot_cost = 0;
    for (size_t i = 0; i < ocp_handler_->getSize(); i++)
    {
      CostStack *cs = ocp_handler_->getCostStack(i);
      QuadraticStateCost *qsc = cs->getComponent<QuadraticStateCost>("state_cost");
      auto data_qsc = qsc->createData();
      qsc->evaluate(xs_[i + 1], us_[i], *data_qsc);
      state_cost += data_qsc->value_;

      QuadraticControlCost *qcc = cs->getComponent<QuadraticControlCost>("control_cost");
      auto data_qcc = qcc->createData();
      qcc->evaluate(xs_[i + 1], us_[i], *data_qcc);
      control_cost += data_qcc->value_;

      for (auto const &name : ocp_handler_->getModelHandler().getFeetNames())
      {
        QuadraticResidualCost *qrc = cs->getComponent<QuadraticResidualCost>(name + "_pose_cost");
        auto data_qrc = qrc->createData();
        qrc->evaluate(xs_[i + 1], us_[i], *data_qrc);
        foot_cost += data_qrc->value_;

        // FrameTranslationResidual *cfr = qrc->getResidual<FrameTranslationResidual>();
        // auto data_cfr = cfr->createData();
        // cfr->evaluate(xs_[i + 1], *data_cfr);
        // if (name == ocp_handler_->getModelHandler().getFeetNames()[0])
        // {
        //   std::cout << data_cfr->value_.transpose() << std::endl;
        // }
      }
    }

    std::cout << "State cost: " << state_cost << std::endl;
    std::cout << "Control cost: " << control_cost << std::endl;
    std::cout << "Foot cost: " << foot_cost << std::endl;
    std::cout << "Total cost: " << state_cost + control_cost + foot_cost << std::endl;
    std::cout << "Total cost2: " << solver_->workspace_.problem_data.cost_ << std::endl;
  }

} // namespace simple_mpc
