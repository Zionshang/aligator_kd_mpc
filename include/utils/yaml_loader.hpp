#pragma once

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <vector>

/**
 * 结构体用于存储YAML配置文件中的参数
 */
struct YamlParams
{
    // OCP settings
    Eigen::VectorXd w_basepos;
    Eigen::VectorXd w_basevel;
    Eigen::VectorXd w_legpos;
    Eigen::VectorXd w_legvel;
    Eigen::VectorXd w_force;
    Eigen::VectorXd w_foot;
    Eigen::VectorXd w_legacc;

    // MPC settings
    int horizon;
    double timestep;
    int max_iter;
    int num_threads;

    // Dynamic settings
    double friction;

    YamlParams(const std::string &filepath)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(filepath);

            // 读取OCP settings
            w_basepos = yamlSequenceToEigen(config["w_basepos"]);
            w_basevel = yamlSequenceToEigen(config["w_basevel"]);
            w_legpos = yamlSequenceToEigen(config["w_legpos"]);
            w_legvel = yamlSequenceToEigen(config["w_legvel"]);
            w_force = yamlSequenceToEigen(config["w_force"]);
            w_foot = yamlSequenceToEigen(config["w_foot"]);
            w_legacc = yamlSequenceToEigen(config["w_legacc"]);
            
            // 读取MPC settings
            horizon = config["horizon"].as<int>();
            timestep = config["timestep"].as<double>();
            max_iter = config["max_iter"].as<int>();
            num_threads = config["num_threads"].as<int>();
            
            // 读取Dynamic settings
            friction = config["friction"].as<double>();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        }
    }

    /**
     * 将YAML序列转换为Eigen::VectorXd
     * @param node YAML序列节点
     * @return 包含序列值的Eigen向量
     */
    Eigen::VectorXd yamlSequenceToEigen(const YAML::Node &node)
    {
        if (!node.IsSequence())
        {
            throw std::runtime_error("YAML node is not a sequence");
        }

        Eigen::VectorXd vec(node.size());
        for (size_t i = 0; i < node.size(); ++i)
        {
            vec(i) = node[i].as<double>();
        }
        return vec;
    }

    void printParams() const
    {
        std::cout << "===== Weight Settings =====" << std::endl;
        std::cout << "w_basepos: " << w_basepos.transpose() << std::endl;
        std::cout << "w_basevel: " << w_basevel.transpose() << std::endl;
        std::cout << "w_legpos: " << w_legpos.transpose() << std::endl;
        std::cout << "w_legvel: " << w_legvel.transpose() << std::endl;
        std::cout << "w_force: " << w_force.transpose() << std::endl;
        std::cout << "w_foot: " << w_foot.transpose() << std::endl;
        std::cout << "w_legacc: " << w_legacc.transpose() << std::endl;
        
        std::cout << "\n===== MPC Settings =====" << std::endl;
        std::cout << "horizon: " << horizon << std::endl;
        std::cout << "timestep: " << timestep << std::endl;
        std::cout << "max_iter: " << max_iter << std::endl;
        
        std::cout << "\n===== Dynamic Settings =====" << std::endl;
        std::cout << "friction: " << friction << std::endl;
    }
};
