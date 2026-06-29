#pragma once

#include <string>
#include <vector>

#include "Types.h"

struct DatasetFrame
{
    int step;

    double trueX;
    double trueY;
    double trueTheta;
    
    double velocity;
    double angularVelocity;

    std::vector<Observation> observations;
};

class DatasetLoader
{
public:

    bool load(const std::string& filename);

    bool hasNext() const;

    DatasetFrame nextFrame();

private:

    std::vector<DatasetFrame> frames;

    size_t currentFrame = 0;
};