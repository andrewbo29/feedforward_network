#include <stdexcept>
#include <iostream>
#include "FullyConnectedLayer.h"
#include "Tanh.h"
#include "Sigmoid.h"

FullyConnectedLayer::FullyConnectedLayer(int size, string activation_type) {
    for (int i = 0; i < size; ++i) {
        neurons.push_back(Neuron());

        if (activation_type == "Sigmoid") {
//            activations.push_back(Sigmoid());
            throw runtime_error("No Sigmoid ");
        } else if (activation_type == "Tanh") {
            activations.push_back(Tanh());
        }
        else {
            throw runtime_error("No such activation function: " + activation_type);
        }
    }
}

vector<double> FullyConnectedLayer::forward(vector<double> &input) {
    vector<double> new_input(input);
    new_input.push_back(1);
    vector<double> res;

    for (size_t i = 0; i < neurons.size(); ++i) {
        res.push_back(activations[i].forward(neurons[i].forward(new_input)));
//        res.push_back(activations[i].forward(neurons[i].forward(input)));
    }

    return res;
}

void FullyConnectedLayer::backward(vector<vector<double>> &weights, vector<double> &deltas) {
    if (!weights.empty()) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            vector<double> neuron_weights;
            for (auto &w : weights) {
                neuron_weights.push_back(w[i]);
            }
            neurons[i].backward(neuron_weights, deltas, activations[i].backward());
        }
    } else {
        for (size_t i = 0; i < neurons.size(); ++i) {
            vector<double> neuron_weights;
            neurons[i].backward(neuron_weights, deltas, activations[i].backward());
        }
    }
}

vector<vector<double>> FullyConnectedLayer::get_weights() {
    vector<vector<double>> all_weights;
    for (auto &neuron : neurons) {
        all_weights.push_back(neuron.get_weights());
    }
    return all_weights;
}

vector<double> FullyConnectedLayer::get_deltas() {
    vector<double> deltas;
    for (auto &neuron : neurons) {
        deltas.push_back(neuron.get_delta());
    }
    return deltas;
}

void FullyConnectedLayer::update_weights(double learning_rate) {
    for (size_t i = 0; i < neurons.size(); ++i) {
//        cout << "   Neuron " << i << ":" << endl;
        neurons[i].update_weights(learning_rate);
    }
}











