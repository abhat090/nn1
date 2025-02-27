cmake_minimum_required(VERSION 3.28)
project(nn1)

set(EIGEN_DIR /opt/homebrew/Cellar/eigen/3.4.0_1)
set(QT_DIR /opt/homebrew/Cellar/qt/6.8.2)
include_directories(${EIGEN_DIR}/include/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/nn/inc)

include_directories(${QT_DIR}/include)
link_directories(${QT_DIR}/lib)

set(CMAKE_CXX_STANDARD 23)

add_executable(nn1 main.cpp
        mnist/mnist_reader.h
        mnist/mnist_reader.cpp
        nn/inc/util.h
        nn/inc/layer.h
        nn/src/layer.cpp
        nn/inc/network.h
        nn/src/network.cpp)
