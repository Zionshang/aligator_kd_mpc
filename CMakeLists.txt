cmake_minimum_required(VERSION 3.10.0)
project(aligator_kd_mpc VERSION 0.1.0 LANGUAGES C CXX)

find_package(pinocchio REQUIRED)
find_package(aligator REQUIRED)
find_package(example-robot-data REQUIRED)
find_package(ndcurves REQUIRED)
include_directories(include)

file(GLOB_RECURSE SRC_LIST 
    ./src/*.cpp
)
add_executable(aligator_kd_mpc ${SRC_LIST})
target_link_libraries(aligator_kd_mpc pinocchio::pinocchio)
target_link_libraries(aligator_kd_mpc aligator::aligator)
target_link_libraries(aligator_kd_mpc example-robot-data::example-robot-data)
target_link_libraries(aligator_kd_mpc ndcurves::ndcurves)

configure_file (
  "${PROJECT_SOURCE_DIR}/include/utils/project_path.hpp.in"
  "${PROJECT_SOURCE_DIR}/include/utils/project_path.hpp" @ONLY
)