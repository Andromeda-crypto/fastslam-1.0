// FastSLAM1.0.cpp
// FastSLAM 1.0 using range + bearing observations
// Includes synthetic simulator test loop and RMSE evaluation

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <Eigen/Dense>

#include "Types.h"
#include "Simulator.h"

constexpr double PI = std::acos(-1.0);

struct Landmark {
    Eigen::Vector2d mean;
    Eigen::Matrix2d covariance;
    bool observed = false;

    Landmark() {
        mean << 0.0, 0.0;
        covariance = Eigen::Matrix2d::Identity() * 1000.0;
    }
};

struct Particle {
    Eigen::Vector3d pose;
    std::vector<Landmark> landmarks;
    double weight = 1.0;
};

class FastSLAM {
public:
    FastSLAM(int num_particles, int num_landmarks)
        : num_particles(num_particles),
          num_landmarks(num_landmarks),
          rng(std::random_device{}()) {

        particles.resize(num_particles);

        for (auto& particle : particles) {
            particle.pose << 0.0, 0.0, 0.0;
            particle.landmarks.resize(num_landmarks);
            particle.weight = 1.0 / num_particles;
        }
    }

    void predict(double velocity, double angular_velocity, double dt) {
        std::normal_distribution<double> noise_v(0.0, motion_noise_v);
        std::normal_distribution<double> noise_w(0.0, motion_noise_w);

        for (auto& particle : particles) {
            double v = velocity + noise_v(rng);
            double w = angular_velocity + noise_w(rng);

            double theta = particle.pose(2);

            particle.pose(0) += v * std::cos(theta) * dt;
            particle.pose(1) += v * std::sin(theta) * dt;
            particle.pose(2) += w * dt;

            normalizeAngle(particle.pose(2));
        }
    }

    void update(const std::vector<Observation>& observations) {
        Eigen::Matrix2d R;
        R << range_noise * range_noise, 0.0,
             0.0, bearing_noise * bearing_noise;

        for (auto& particle : particles) {
            for (const auto& obs : observations) {
                if (obs.landmark_id < 0 || obs.landmark_id >= num_landmarks) {
                    continue;
                }

                Landmark& lm = particle.landmarks[obs.landmark_id];

                if (!lm.observed) {
                    initializeLandmark(particle, lm, obs);
                } else {
                    updateLandmark(particle, lm, obs, R);
                }
            }
        }

        normalizeWeights();
    }

    void resample() {
        std::vector<Particle> new_particles;
        new_particles.reserve(num_particles);

        std::vector<double> weights;
        weights.reserve(num_particles);

        for (const auto& particle : particles) {
            weights.push_back(particle.weight);
        }

        std::discrete_distribution<int> dist(weights.begin(), weights.end());

        for (int i = 0; i < num_particles; i++) {
            Particle selected = particles[dist(rng)];
            selected.weight = 1.0 / num_particles;
            new_particles.push_back(selected);
        }

        particles = new_particles;
    }

    const std::vector<Particle>& getParticles() const {
        return particles;
    }

        Eigen::Vector3d estimatePoseMean() const {
        Eigen::Vector3d estimate;
        estimate << 0.0, 0.0, 0.0;

        for (const auto& particle: particles) {
            estimate += particle.pose;
        }

        return estimate / static_cast<double>(particles.size());
    }
    
    double effectiveSampleSize() const {
        double sum_squared_weights = 0.0;
        
        for (const auto& particle: particles) {
            sum_squared_weights += particle.weight * particle.weight;
        }

        if (sum_squared_weights <= 1e-12){
            return 0.0;
        }
        return 1.0/sum_squared_weights;
    }



private:
    int num_particles;
    int num_landmarks;
    std::vector<Particle> particles;

    std::mt19937 rng;

    double motion_noise_v = 0.1;
    double motion_noise_w = 0.05;

    double range_noise = 0.2;
    double bearing_noise = 0.05;

    void initializeLandmark(
        const Particle& particle,
        Landmark& lm,
        const Observation& obs
    ) {
        double global_bearing = particle.pose(2) + obs.bearing;

        lm.mean(0) = particle.pose(0) + obs.range * std::cos(global_bearing);
        lm.mean(1) = particle.pose(1) + obs.range * std::sin(global_bearing);

        Eigen::Matrix2d R;
        R << range_noise * range_noise, 0.0,
             0.0, bearing_noise * bearing_noise;

        double c = std::cos(global_bearing);
        double s = std::sin(global_bearing);

        Eigen::Matrix2d Gz;
        Gz << c, -obs.range * s,
              s,  obs.range * c;

        lm.covariance = Gz * R * Gz.transpose();
        lm.observed = true;
    }

    void updateLandmark(
        Particle& particle,
        Landmark& lm,
        const Observation& obs,
        const Eigen::Matrix2d& R
    ) {
        double dx = lm.mean(0) - particle.pose(0);
        double dy = lm.mean(1) - particle.pose(1);

        double q = dx * dx + dy * dy;

        if (q < 1e-9) {
            return;
        }

        double sqrt_q = std::sqrt(q);

        Eigen::Vector2d expected;
        expected(0) = sqrt_q;
        expected(1) = std::atan2(dy, dx) - particle.pose(2);
        normalizeAngle(expected(1));

        Eigen::Vector2d z;
        z << obs.range, obs.bearing;

        Eigen::Vector2d innovation = z - expected;
        normalizeAngle(innovation(1));

        Eigen::Matrix2d H;
        H << dx / sqrt_q, dy / sqrt_q,
            -dy / q,      dx / q;

        Eigen::Matrix2d S = H * lm.covariance * H.transpose() + R;
        Eigen::Matrix2d K = lm.covariance * H.transpose() * S.inverse();

        lm.mean = lm.mean + K * innovation;
        lm.covariance = (Eigen::Matrix2d::Identity() - K * H) * lm.covariance;

        particle.weight *= measurementLikelihood(innovation, S);
    }

    void normalizeWeights() {
        double total = 0.0;

        for (const auto& particle : particles) {
            total += particle.weight;
        }

        if (total <= 1e-12) {
            for (auto& particle : particles) {
                particle.weight = 1.0 / num_particles;
            }
            return;
        }

        for (auto& particle : particles) {
            particle.weight /= total;
        }
    }

    double measurementLikelihood(
        const Eigen::Vector2d& innovation,
        const Eigen::Matrix2d& S
    ) {
        double det = S.determinant();

        if (det <= 0.0) {
            return 1e-12;
        }

        double exponent =
            -0.5 * (innovation.transpose() * S.inverse() * innovation)(0, 0);

        double normalizer = 1.0 / (2.0 * PI * std::sqrt(det));

        return std::max(normalizer * std::exp(exponent), 1e-12);
    }

    void normalizeAngle(double& angle) {
        while (angle > PI) {
            angle -= 2.0 * PI;
        }

        while (angle < -PI) {
            angle += 2.0 * PI;
        }
    }

};

int main() {
    constexpr int NUM_STEPS = 50;

    FastSLAM slam(100, 3);
    Simulator simulator;

    double total_position_error_squared = 0.0;

    for (int t = 0; t < NUM_STEPS; t++) {
        double velocity = 1.0;
        double angular_velocity = 0.05;
        double dt = 0.1;

        simulator.move(velocity, angular_velocity, dt);

        std::vector<Observation> observations =
            simulator.generateObservations();

        slam.predict(velocity, angular_velocity, dt);
        slam.update(observations);
        double n_eff = slam.effectiveSampleSize();
        if (n_eff < 80.0){
            slam.resample();
        }


        const auto& particles = slam.getParticles();
        Eigen::Vector3d true_pose = simulator.getTruePose();

        Eigen::Vector3d estimate = slam.estimatePoseMean();

        double dx = estimate(0) - true_pose(0);
        double dy = estimate(1) - true_pose(1);
        double position_error = std::sqrt(dx * dx + dy * dy);

        total_position_error_squared += position_error * position_error;

        std::cout << "Step " << t << "\n";
        std::cout << "True pose:\n" << true_pose << "\n";
        std::cout << "Estimated mean pose:\n" << estimate << "\n";
        std::cout << "Position error: " << position_error << "\n\n";
        std::cout <<"Effective Sample Size: "<< n_eff << "\n\n";
    }

    double trajectory_rmse =
        std::sqrt(total_position_error_squared / static_cast<double>(NUM_STEPS));

    std::cout << "Trajectory RMSE: " << trajectory_rmse << "\n";

    return 0;
}