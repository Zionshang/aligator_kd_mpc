#pragma once

#include <webots/Motor.hpp>
#include <webots/PositionSensor.hpp>
#include <webots/InertialUnit.hpp>
#include <webots/Gyro.hpp>
#include <webots/Supervisor.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class WebotsInterface
{
public:
    WebotsInterface();
    ~WebotsInterface();
    void recvState(Eigen::VectorXd &state_vector); // second
    void sendCmd(const Eigen::VectorXd &tau);
    bool isRunning();
    double current_time() { return current_time_; }
    void resetSim() { supervisor_->simulationReset(); }

private:
    void initRecv();
    void initSend();

    int time_step_;
    double current_time_;

    Eigen::VectorXd last_q_;

    // webots interface
    webots::Supervisor *supervisor_;
    webots::Node *robot_node_;
    webots::Motor *joint_motor_[12];
    webots::PositionSensor *joint_sensor_[12];
    webots::InertialUnit *imu_;
    webots::Gyro *gyro_;

    std::string robot_name_ = "GalileoMini";
    std::string imu_name_ = "imu";
    std::string gyro_name_ = "gyro";
    std::vector<std::string> joint_sensor_name_ = {"FL_abd_joint_sensor", "FL_hip_joint_sensor", "FL_knee_joint_sensor",
                                                   "FR_abd_joint_sensor", "FR_hip_joint_sensor", "FR_knee_joint_sensor",
                                                   "HL_abd_joint_sensor", "HL_hip_joint_sensor", "HL_knee_joint_sensor",
                                                   "HR_abd_joint_sensor", "HR_hip_joint_sensor", "HR_knee_joint_sensor"};
    std::vector<std::string> joint_motor_name_ = {"FL_abd_joint", "FL_hip_joint", "FL_knee_joint",
                                                  "FR_abd_joint", "FR_hip_joint", "FR_knee_joint",
                                                  "HL_abd_joint", "HL_hip_joint", "HL_knee_joint",
                                                  "HR_abd_joint", "HR_hip_joint", "HR_knee_joint",};
};