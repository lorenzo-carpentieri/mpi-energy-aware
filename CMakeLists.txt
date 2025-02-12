cmake_minimum_required(VERSION 3.17)
project(MpiEnergyAware VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(Het_GPU_Power_Measurement/)  # Include the external project

add_executable(ar_aware src/ar_aware.cpp)

target_link_libraries(ar_aware PRIVATE Het_GPU_Power_Measurement)

