cmake_minimum_required(VERSION 3.10.0)
project(aligator_kd_mpc VERSION 0.1.0 LANGUAGES C CXX)

find_package(pinocchio REQUIRED)
find_package(aligator REQUIRED)
find_package(example-robot-data REQUIRED)
find_package(ndcurves REQUIRED)
include_directories(include)


# webots
set(WEBOTS_HOME "/usr/local/webots")
link_directories(${WEBOTS_HOME}/lib/controller)
set(WEBOTSLIBRARIES m ${CMAKE_SHARED_LIBRARY_PREFIX}Controller${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_PREFIX}CppController${CMAKE_SHARED_LIBRARY_SUFFIX})
include_directories(${WEBOTS_HOME}/include/controller/c ${WEBOTS_HOME}/include/controller/cpp)

file(GLOB_RECURSE SRC_LIST
  src/foot-trajectory.cpp
  src/kinodynamics.cpp
  src/mpc.cpp
  src/ocp-handler.cpp
  src/robot-handler.cpp
  src/webots_interface.cpp
)
add_executable(aligator_kd_mpc ${SRC_LIST} src/main.cpp)
target_link_libraries(aligator_kd_mpc pinocchio::pinocchio)
target_link_libraries(aligator_kd_mpc aligator::aligator)
target_link_libraries(aligator_kd_mpc example-robot-data::example-robot-data)
target_link_libraries(aligator_kd_mpc ndcurves::ndcurves)
target_link_libraries(aligator_kd_mpc ${WEBOTSLIBRARIES})

add_executable(test ${SRC_LIST} test.cpp)
target_link_libraries(test pinocchio::pinocchio)
target_link_libraries(test aligator::aligator)
target_link_libraries(test example-robot-data::example-robot-data)
target_link_libraries(test ndcurves::ndcurves)
target_link_libraries(test ${WEBOTSLIBRARIES})

configure_file(
  "${PROJECT_SOURCE_DIR}/include/utils/project_path.hpp.in"
  "${PROJECT_SOURCE_DIR}/include/utils/project_path.hpp" @ONLY
)
