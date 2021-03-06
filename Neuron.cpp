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
            dw.push_back(0);
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
    double s = 0;
    if (!w.empty()) {
        for (size_t i = 0; i < d.size(); ++i) {
            s += w[i] * d[i];
        }
    } else {
        for (size_t i = 0; i < d.size(); ++i) {
            s += d[i];
        }
    }
    delta = s * dActive;

    for (size_t i = 0; i < weights.size(); ++i) {
        dw[i] += x[i] * delta;
    }
}

void Neuron::update_weights(double learning_rate) {
//    cout << "       Delta: " << delta << endl;
//    cout << "       Weights: ";
    for (size_t i = 0; i < weights.size(); ++i) {
//        weights[i] -= learning_rate * x[i] * delta;
        weights[i] -= learning_rate * dw[i];
        dw[i] = 0;
//        cout << weights[i] << " ";
    }
//    cout << endl;
}

vector<double> Neuron::get_grad() {
    vector<double> grad;
    for (size_t i = 0; i < weights.size(); ++i) {
        grad.push_back(x[i] * delta);
    }
    return grad;
}






