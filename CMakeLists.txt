cmake_minimum_required(VERSION 2.8)
project(feedforward_network)

find_package(OpenCV REQUIRED)
message(STATUS ${OpenCV_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(GTEST_SOURCE googletest/googletest/src/gtest-all.cc googletest/googletest/src/gtest_main.cc)
include_directories(googletest/googletest)
include_directories(googletest/googletest/include)

add_library(gtest_lib STATIC ${GTEST_SOURCE})

set(FFN_SOURCE_FILES
        Neuron.cpp
        Neuron.h
        InputLayer.cpp
        InputLayer.h
        FullyConnectedLayer.cpp
        FullyConnectedLayer.h
        FeedForwardNet.cpp
        FeedForwardNet.h
        imageProcessing.cpp
        imageProcessing.h
        EuclideanLoss.cpp
        EuclideanLoss.h
        Sigmoid.cpp
        Sigmoid.h
        Tanh.cpp
        Tanh.h
        Activation.h
        EuclideanLoss.cpp
        EuclideanLoss.h
        )

add_library(FFN STATIC ${FFN_SOURCE_FILES})

add_executable(feedforward_network main.cpp)
target_link_libraries(feedforward_network FFN ${OpenCV_LIBS} -pthread)

add_executable(feedforward_network_gtest test.cpp)
target_link_libraries(feedforward_network_gtest FFN ${OpenCV_LIBS} gtest_lib -pthread)
