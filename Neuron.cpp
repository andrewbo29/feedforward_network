#include <cstdlib>
#include <random>
#include <chrono>
#include <iostream>
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
            weights.push_back(0.01 * distribution(generator));
        }
    }

    double output = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        output += weights[i] * input[i];
    }

    x = input;

    return output;
}

void Neuron::backward(vector<double> &w, vector<double> &d, double dActive) {
    if (!w.empty()) {
        double s = 0;
        for (size_t i = 0; i < d.size(); ++i) {
            s += w[i] * d[i];
        }
        delta = s * dActive;
    } else {
        double s = 0;
        for (size_t i = 0; i < d.size(); ++i) {
            s += d[i];
        }
        delta = s * dActive;
    }
}

void Neuron::update_weights(double learning_rate) {
//    cout << "       Delta: " << delta << endl;
//    cout << "       Weights: ";
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * x[i] * delta;
//        cout << weights[i] << " ";
    }
//    cout << endl;
}




