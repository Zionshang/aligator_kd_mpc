#include "webots_interface.hpp"

WebotsInterface::WebotsInterface()
{
    supervisor_ = new webots::Supervisor();
    time_step_ = (int)supervisor_->getBasicTimeStep();
    std::cout << "timeStep in simulation is :" << time_step_ << std::endl;

    initRecv();
    initSend();

    last_q_.resize(joint_sensor_name_.size());
    last_q_ << 0, 0.72, -1.44,
        0, 0.72, -1.44,
        0, 0.72, -1.44,
        0, 0.72, -1.44;
}

WebotsInterface::~WebotsInterface()
{
    delete supervisor_;
}

void WebotsInterface::recvState(Eigen::VectorXd &state_vector)
{
    current_time_ = supervisor_->getTime();
    Eigen::VectorXd q(19), v(18);

    // sensor data
    auto imu_data = imu_->getQuaternion();
    Eigen::Quaterniond quaternion(imu_data[3], imu_data[0], imu_data[1], imu_data[2]);        // x,y,z,w
    Eigen::Vector3d angular_vel_B = Eigen::Map<const Eigen::Vector3d>(gyro_->getValues());    // expressed in BODY frame
    Eigen::Vector3d robotPos = Eigen::Map<const Eigen::Vector3d>(robot_node_->getPosition()); // expressed in WORLD frame
    Eigen::Vector3d robotVel = Eigen::Map<const Eigen::Vector3d>(robot_node_->getVelocity()); // expressed in WORLD frame
    Eigen::Vector3d robotVel_B = quaternion.toRotationMatrix().transpose() * robotVel;        // expressed in BODY frame

    q.head(6) << robotPos, quaternion.coeffs();
    v.head(6) << robotVel_B, angular_vel_B;

    for (int i = 0; i < 12; i++)
    {
        q(7 + i) = joint_sensor_[i]->getValue();
        v(6 + i) = (q(7 + i) - last_q_(i)) / double(time_step_) * 1000;
        last_q_(i) = q(7 + i);
    }
    state_vector << q, v;
}

void WebotsInterface::recvUserCmd(Eigen::Vector3d &v_cmd)
{
    key_ = keyboard_->getKey();
    if (key_ != last_key_)
    {
        switch (key_)
        {
        case 'w':
        case 'W':
            v_cmd(0) += 0.1;
            break;
        case 'b':
        case 'B':
            v_cmd(0) += -0.1;
            break;
        case 'a':
        case 'A':
            v_cmd(1) += 0.1;
            break;
        case 'd':
        case 'D':
            v_cmd(1) += -0.1;
            break;
        case 's':
        case 'S':
            v_cmd.setZero();
            v_cmd.setZero();
            break;
        }
    }
    last_key_ = key_;
}

void WebotsInterface::sendCmd(const Eigen::VectorXd &tau)
{
    for (int i = 0; i < 12; i++)
    {
        joint_motor_[i]->setTorque(tau(i));
    }
}

bool WebotsInterface::isRunning()
{
    if (supervisor_->step(time_step_) != -1)
        return true;
    else
        return false;
}

void WebotsInterface::initRecv()
{
    // supervisor init
    robot_node_ = supervisor_->getFromDef(robot_name_);
    if (robot_node_ == NULL)
    {
        printf("error supervisor\n");
        exit(1);
    }

    // sensor init
    imu_ = supervisor_->getInertialUnit(imu_name_);
    imu_->enable(time_step_);
    gyro_ = supervisor_->getGyro(gyro_name_);
    gyro_->enable(time_step_);
    for (int i = 0; i < 12; i++)
    {
        joint_sensor_[i] = supervisor_->getPositionSensor(joint_sensor_name_[i]);
        joint_sensor_[i]->enable(time_step_);
    }
    keyboard_ = supervisor_->getKeyboard();
    keyboard_->enable(time_step_);
}

void WebotsInterface::initSend()
{
    for (int i = 0; i < 12; i++)
        joint_motor_[i] = supervisor_->getMotor(joint_motor_name_[i]);
}