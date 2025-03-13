#pragma once
#include <Eigen/Core>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

void saveVectorsToCsv(const std::string &filename, const std::vector<Eigen::VectorXd> &vectors)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw "Unable to open file for writing";
    }

    for (const auto &vec : vectors)
    {
        for (int i = 0; i < vec.size(); ++i)
        {
            file << vec(i);
            if (i < vec.size() - 1)
                file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

std::vector<Eigen::VectorXd> readVectorsFromCsv(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw "Unable to open file for reading";
    }
    
    std::vector<Eigen::VectorXd> result;
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        std::vector<double> data;
        std::string token;
        while (std::getline(ss, token, ','))
        {
            data.push_back(std::stod(token));
        }
        Eigen::VectorXd vec(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            vec[i] = data[i];
        }
        result.push_back(vec);
    }
    file.close();
    return result;
}