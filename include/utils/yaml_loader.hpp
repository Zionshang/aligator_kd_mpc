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
    Eigen::VectorXd w_basepos;
    Eigen::VectorXd w_basevel;
    Eigen::VectorXd w_legpos;
    Eigen::VectorXd w_legvel;
    Eigen::VectorXd w_force;
    Eigen::VectorXd w_foot;
    Eigen::VectorXd w_legacc;

    YamlParams(const std::string &filepath)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(filepath);

            w_basepos = yamlSequenceToEigen(config["w_basepos"]);
            w_basevel = yamlSequenceToEigen(config["w_basevel"]);
            w_legpos = yamlSequenceToEigen(config["w_legpos"]);
            w_legvel = yamlSequenceToEigen(config["w_legvel"]);
            w_force = yamlSequenceToEigen(config["w_force"]);
            w_foot = yamlSequenceToEigen(config["w_foot"]);
            w_legacc = yamlSequenceToEigen(config["w_legacc"]);
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
        std::cout << "w_basepos: " << w_basepos.transpose() << std::endl;
        std::cout << "w_basevel: " << w_basevel.transpose() << std::endl;
        std::cout << "w_legpos: " << w_legpos.transpose() << std::endl;
        std::cout << "w_legvel: " << w_legvel.transpose() << std::endl;
        std::cout << "w_force: " << w_force.transpose() << std::endl;
        std::cout << "w_foot: " << w_foot.transpose() << std::endl;
        std::cout << "w_legacc: " << w_legacc.transpose() << std::endl;
    }
};
