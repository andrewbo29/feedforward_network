cmake_minimum_required(VERSION 3.4)
project(feedforward_network)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Neuron.cpp Neuron.h Sigmoid.cpp Sigmoid.h InputLayer.cpp InputLayer.h FullyConnectedLayer.cpp FullyConnectedLayer.h FeedForwardNet.cpp FeedForwardNet.h)
add_executable(feedforward_network ${SOURCE_FILES})