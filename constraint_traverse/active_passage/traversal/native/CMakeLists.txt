cmake_minimum_required(VERSION 3.18)
project(active_passage_smac LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(ament_cmake REQUIRED)
find_package(nav2_smac_planner REQUIRED)
find_package(Python 3.11 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_smac_lattice MODULE smac_lattice.cpp)
target_link_options(_smac_lattice PRIVATE "-Wl,--disable-new-dtags")
ament_target_dependencies(_smac_lattice PUBLIC nav2_smac_planner)
set_target_properties(_smac_lattice PROPERTIES
  BUILD_RPATH "/opt/ros/humble/lib;/opt/ros/humble/lib/x86_64-linux-gnu"
  INSTALL_RPATH "/opt/ros/humble/lib;/opt/ros/humble/lib/x86_64-linux-gnu")
