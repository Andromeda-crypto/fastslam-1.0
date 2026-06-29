#include <fstream>
#include <iostream>
#include <vector>

#include "Simulator.h"
#include "Types.h"

int main() {
    Simulator simulator;

    std::ofstream file("data/synthetic/synthetic_dataset.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to create synthetic dataset file.\n";
        return 1;
    }

    file << "step,true_x,true_y,true_theta,velocity,angular_velocity,landmark_id,range,bearing\n";

    constexpr int NUM_STEPS = 50;
    constexpr double velocity = 1.0;
    constexpr double angular_velocity = 0.05;
    constexpr double dt = 0.1;

    for (int step = 0; step < NUM_STEPS; step++) {
        simulator.move(velocity, angular_velocity, dt);
        Eigen::Vector3d true_pose = simulator.getTruePose();

        std::vector<Observation> observations =
            simulator.generateObservations();

        for (const auto& obs : observations) {
            file << step << ","
                << true_pose(0) << ","
                << true_pose(1) << ","
                << true_pose(2) << ","
                << velocity << ","
                << angular_velocity << ","
                << obs.landmark_id << ","
                << obs.range << ","
                << obs.bearing << "\n";
        }
    }

    file.close();

    std::cout << "Saved synthetic dataset to data/synthetic/synthetic_dataset.csv\n";

    return 0;
}