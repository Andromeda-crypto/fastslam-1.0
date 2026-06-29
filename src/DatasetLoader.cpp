#include "DatasetLoader.h"

#include <fstream>
#include <sstream>
#include <iostream>

bool DatasetLoader::load(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open dataset: " << filename << "\n";
        return false;
    }

    frames.clear();
    currentFrame = 0;

    std::string line;
    std::getline(file, line); // header

    int currentStep = -1;
    DatasetFrame frame;

    while (std::getline(file, line)) {
        std::stringstream ss(line);

        std::string token;
        std::vector<std::string> values;

        while (std::getline(ss, token, ',')) {
            values.push_back(token);
        }

        if (values.size() != 9) {
            continue;
        }

        int step = std::stoi(values[0]);

        double trueX = std::stod(values[1]);
        double trueY = std::stod(values[2]);
        double trueTheta = std::stod(values[3]);

        double velocity = std::stod(values[4]);
        double angularVelocity = std::stod(values[5]);

        int landmarkId = std::stoi(values[6]);
        double range = std::stod(values[7]);
        double bearing = std::stod(values[8]);

        if (step != currentStep) {
            if (currentStep != -1) {
                frames.push_back(frame);
            }

            currentStep = step;

            frame = DatasetFrame{};
            frame.step = step;
            frame.trueX = trueX;
            frame.trueY = trueY;
            frame.trueTheta = trueTheta;
            frame.velocity = velocity;
            frame.angularVelocity = angularVelocity;
        }

        frame.observations.push_back({landmarkId, range, bearing});
    }

    if (currentStep != -1) {
        frames.push_back(frame);
    }

    return !frames.empty();
}

bool DatasetLoader::hasNext() const {
    return currentFrame < frames.size();
}

DatasetFrame DatasetLoader::nextFrame() {
    return frames[currentFrame++];
}