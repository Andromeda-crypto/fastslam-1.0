#include "Simulator.h"

#include <cmath>

constexpr double PI = std::acos(-1.0);

Simulator::Simulator()
    : rng(std::random_device{}()) {
    true_pose << 0.0, 0.0, 0.0;
}

void Simulator::move(double velocity, double angular_velocity, double dt) {
    true_pose(0) += velocity * std::cos(true_pose(2)) * dt;
    true_pose(1) += velocity * std::sin(true_pose(2)) * dt;
    true_pose(2) += angular_velocity * dt;

    normalizeAngle(true_pose(2));
}

std::vector<Observation> Simulator::generateObservations() {
    std::vector<Observation> observations;

    std::normal_distribution<double> range_dist(0.0, range_noise);
    std::normal_distribution<double> bearing_dist(0.0, bearing_noise);

    for (int i = 0; i < static_cast<int>(world.landmarks.size()); i++) {
        double dx = world.landmarks[i](0) - true_pose(0);
        double dy = world.landmarks[i](1) - true_pose(1);

        double range = std::sqrt(dx * dx + dy * dy);
        double bearing = std::atan2(dy, dx) - true_pose(2);

        normalizeAngle(bearing);

        if (range < max_range) {
            range += range_dist(rng);
            bearing += bearing_dist(rng);

            normalizeAngle(bearing);

            observations.push_back({i, range, bearing});
        }
    }

    return observations;
}

Eigen::Vector3d Simulator::getTruePose() const {
    return true_pose;
}

void Simulator::normalizeAngle(double& angle) {
    while (angle > PI) {
        angle -= 2.0 * PI;
    }

    while (angle < -PI) {
        angle += 2.0 * PI;
    }
}