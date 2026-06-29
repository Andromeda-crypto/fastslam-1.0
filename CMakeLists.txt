cmake_minimum_required(VERSION 3.16)
project(FastSLAM_1_0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
include_directories("C:/msys64/mingw64/include/eigen3")

add_executable(fastslam
    src/FastSLAM1.0.cpp
    src/DatasetLoader.cpp
)

add_executable(generate_dataset
    src/GenerateSyntheticDataset.cpp
    src/Simulator.cpp
)