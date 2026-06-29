#pragma once

#include <vector>
#include <Eigen/Dense>

struct World {
    std::vector<Eigen::Vector2d> landmarks;

    World() {
        landmarks.push_back(Eigen::Vector2d(5.0, 5.0));
        landmarks.push_back(Eigen::Vector2d(10.0, 0.0));
        landmarks.push_back(Eigen::Vector2d(5.0, -5.0));

    }
};