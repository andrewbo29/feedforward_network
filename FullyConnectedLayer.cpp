#include "FullyConnectedLayer.h"

vector<double> FullyConnectedLayer::forward(vector<double> input) {
    vector<double> res;

    for (decltype(neurons.size()) i = 0; i < neurons.size(); ++i) {
        res.push_back(activasions[i].forward(neurons[i].forward(input)));
    }

    return res;
}

FullyConnectedLayer::FullyConnectedLayer(int size) {
    for (int i = 0; i < size; ++i) {
        neurons.push_back(Neuron());
        activasions.push_back(Sigmoid());
    }
}



