cmake_minimum_required(VERSION 3.4)
project(feedforward_network)

find_package( OpenCV REQUIRED )
message(STATUS ${OpenCV_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Neuron.cpp Neuron.h Sigmoid.cpp Sigmoid.h InputLayer.cpp InputLayer.h FullyConnectedLayer.cpp FullyConnectedLayer.h FeedForwardNet.cpp FeedForwardNet.h imageProcessing.cpp imageProcessing.h)
add_executable(feedforward_network ${SOURCE_FILES})
target_link_libraries( feedforward_network ${OpenCV_LIBS} )