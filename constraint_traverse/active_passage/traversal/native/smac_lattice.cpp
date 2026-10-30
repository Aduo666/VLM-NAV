#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_smac_planner/a_star.hpp"
#include "nav2_smac_planner/collision_checker.hpp"
#include "nav2_smac_planner/node_lattice.hpp"

namespace py = pybind11;
using Star = nav2_smac_planner::AStarAlgorithm<nav2_smac_planner::NodeLattice>;
static std::mutex search_mutex;

class LatticePlanner
{
public:
  explicit LatticePlanner(const std::string & path) : path_(path)
  {
    metadata_ = nav2_smac_planner::LatticeMotionTable::getLatticeMetadata(path);
  }

  py::dict plan(
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> costs,
    double resolution, double origin_x, double origin_y,
    const std::vector<double> & start, const std::vector<double> & goal,
    py::array_t<double, py::array::c_style | py::array::forcecast> polygon,
    double time_limit, double tolerance)
  {
    if (costs.ndim() != 2 || costs.shape(0) < 2 || costs.shape(1) < 2 ||
      polygon.ndim() != 2 || polygon.shape(1) != 2 || polygon.shape(0) < 3 ||
      start.size() != 3 || goal.size() != 2 || resolution <= 0 || time_limit <= 0 ||
      std::abs(resolution - metadata_.grid_resolution) > 1e-6)
    {
      throw std::invalid_argument("Invalid Smac input or primitive/grid resolution mismatch");
    }
    const auto started = std::chrono::steady_clock::now();
    std::vector<std::array<double, 3>> points;
    std::string reason = "NO_PATH";
    int total_iterations = 0;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(search_mutex);
      nav2_costmap_2d::Costmap2D map(costs.shape(1), costs.shape(0), resolution, origin_x, origin_y, 255);
      std::memcpy(map.getCharMap(), costs.data(), costs.size());
      unsigned int sx, sy, gx, gy;
      if (!map.worldToMap(start[0], start[1], sx, sy) || !map.worldToMap(goal[0], goal[1], gx, gy)) {
        reason = "OUTSIDE_MAP";
      } else {
        nav2_costmap_2d::Footprint footprint;
        auto vertices = polygon.unchecked<2>();
        for (py::ssize_t i = 0; i < polygon.shape(0); ++i) {
          geometry_msgs::msg::Point point;
          point.x = vertices(i, 0);
          point.y = vertices(i, 1);
          footprint.push_back(point);
        }
        nav2_smac_planner::GridCollisionChecker checker(&map, 72u, nullptr);
        checker.setFootprint(footprint, false, -1.0);
        if (!search_) {
          nav2_smac_planner::SearchInfo info{};
          info.minimum_turning_radius = metadata_.min_turning_radius / resolution;
          info.non_straight_penalty = 1.05f;
          info.change_penalty = 0.05f;
          info.reverse_penalty = 2.0f;
          info.cost_penalty = 2.0f;
          info.retrospective_penalty = 0.015f;
          info.rotation_penalty = 5.0f;
          info.analytic_expansion_ratio = 3.5f;
          info.analytic_expansion_max_length = 3.0 / resolution;
          info.lattice_filepath = path_;
          info.cache_obstacle_heuristic = false;
          info.allow_reverse_expansion = true;
          search_ = std::make_unique<Star>(nav2_smac_planner::MotionModel::STATE_LATTICE, info);
        }
        const double bearing = std::atan2(goal[1] - start[1], goal[0] - start[0]);
        std::vector<unsigned int> headings(metadata_.number_of_headings);
        for (unsigned int i = 0; i < headings.size(); ++i) {headings[i] = i;}
        std::stable_sort(headings.begin(), headings.end(), [&](auto a, auto b) {
          return std::abs(std::remainder(metadata_.heading_angles[a] - bearing, 2 * M_PI)) <
                 std::abs(std::remainder(metadata_.heading_angles[b] - bearing, 2 * M_PI));
        });
        for (auto heading : headings) {
          const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
          const double remaining = time_limit - elapsed;
          if (remaining <= 0) {reason = "SEARCH_TIMEOUT"; break;}
          int maximum = 300000;
          const float lookup = static_cast<float>(2 * static_cast<int>(5.0 / resolution) + 1);
          search_->initialize(false, maximum, 100, remaining, lookup, metadata_.number_of_headings);
          search_->setCollisionChecker(&checker);
          double yaw = std::fmod(start[2], 2 * M_PI);
          if (yaw < 0) {yaw += 2 * M_PI;}
          search_->setStart(sx, sy, nav2_smac_planner::NodeLattice::motion_table.getClosestAngularBin(yaw));
          search_->setGoal(gx, gy, heading);
          nav2_smac_planner::NodeLattice::CoordinateVector path;
          int iterations = 0;
          try {
            const bool found = search_->createPath(path, iterations, tolerance / resolution);
            total_iterations += iterations;
            if (found && path.size() >= 2) {
              for (auto it = path.rbegin(); it != path.rend(); ++it) {
                points.push_back({origin_x + (it->x + 0.5) * resolution,
                                  origin_y + (it->y + 0.5) * resolution, it->theta});
              }
              reason = "PATH_FOUND";
              break;
            }
          } catch (const std::runtime_error & error) {
            reason = error.what();
            if (reason.find("Starting point") != std::string::npos) {break;}
          }
        }
      }
    }
    py::array_t<double> output({static_cast<py::ssize_t>(points.size()), py::ssize_t(3)});
    auto array = output.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(points.size()); ++i) {
      for (int j = 0; j < 3; ++j) {array(i, j) = points[i][j];}
    }
    py::dict result;
    result["poses"] = output;
    result["reason"] = reason;
    result["iterations"] = total_iterations;
    result["planning_sec"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
  }

private:
  std::string path_;
  nav2_smac_planner::LatticeMetadata metadata_;
  std::unique_ptr<Star> search_;
};

PYBIND11_MODULE(_smac_lattice, module)
{
  py::class_<LatticePlanner>(module, "LatticePlanner")
    .def(py::init<const std::string &>())
    .def("plan", &LatticePlanner::plan);
}
