#pragma once

#include <vector>
#include <random>
#include <Eigen/Dense>

#include "Types.h"
#include "World.h"

class Simulator {
public:
    Simulator();

    void move(double velocity, double angular_velocity, double dt);
    std::vector<Observation> generateObservations();

    Eigen::Vector3d getTruePose() const;
    const std::vector<Eigen::Vector2d>& getTrueLandmarks() const;

private:
    Eigen::Vector3d true_pose;
    World world;

    double max_range = 15.0;
    double range_noise = 0.2;
    double bearing_noise = 0.05;

    std::mt19937 rng;

    void normalizeAngle(double& angle);
};