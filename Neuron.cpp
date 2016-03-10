#include <cstdlib>
#include <random>
#include <chrono>
#include "Neuron.h"

Neuron::Neuron() {
    weights.clear();
}

double Neuron::forward(vector<double> &input) {
    if (weights.empty()) {
        int seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator(seed);
        normal_distribution<double> distribution(0, 1);
        for (size_t i = 0; i < input.size(); ++i) {
            weights.push_back(distribution(generator));
        }
    }

    double output = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        output += weights[i] * input[i];
    }

    return output;
}

double Neuron::backward(vector<double> &w, vector<double> &d, double dActive) {
    double s = 0;
    for (size_t i = 0; i < d.size(); ++i) {
        s += w[i] * d[i];
    }

    delta = s * dActive;
    return delta;
}


