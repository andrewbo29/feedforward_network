#include <stdexcept>
#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(int size, string activation_type) {
    for (int i = 0; i < size; ++i) {
        neurons.push_back(Neuron());

        if (activation_type == "Sigmoid") {
            activations.push_back(Sigmoid());
        } else {
            throw runtime_error("No such activation function: " + activation_type);
        }
    }
}

vector<double> FullyConnectedLayer::forward(vector<double> &input) {
    vector<double> res;

    for (size_t i = 0; i < neurons.size(); ++i) {
        res.push_back(activations[i].forward(neurons[i].forward(input)));
    }

    return res;
}

void FullyConnectedLayer::backward(vector<vector<double>> &weights, vector<double> &deltas) {
    for (size_t i = 0; i < neurons.size(); ++i) {
        neurons[i].backward(weights[i], deltas, activations[i].backward());
    }
}





