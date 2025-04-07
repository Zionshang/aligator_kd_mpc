#pragma once
#include <Eigen/Core>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

void saveVectorsToCsv(const std::string &filename, const std::vector<Eigen::VectorXd> &vectors);

std::vector<Eigen::VectorXd> readVectorsFromCsv(const std::string &filename);