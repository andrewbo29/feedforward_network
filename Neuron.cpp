#include <cstdlib>
#include "Neuron.h"

double Neuron::forward(vector<double> input) {
    if (weights.empty()) {
        for (decltype(input.size()) i = 0; i < input.size(); ++i) {
            double init_weight = static_cast<double>(rand()) / (RAND_MAX);
            weights.push_back(init_weight);
        }
    }

    double output = 0;
    for (decltype(input.size()) i = 0; i < input.size(); ++i) {
        output += weights[i] * input[i];
    }

    return output;
}

double Neuron::backward(vector<double> w, vector<double> d, double dActive) {
    double s = 0;
    for (decltype(d.size()) i = 0; i < d.size(); ++i) {
        s += w[i] * d[i];
    }

    delta = s * dActive;
    return delta;
}

Neuron::Neuron() {
    weights.clear();
}


