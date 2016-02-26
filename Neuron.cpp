#include "Neuron.h"

double Neuron::forward(vector<double> input) {
    values = input;

    double output = 0;
    for (decltype(input.size()) i = 0; i < input.size(); ++i) {
        output += weights[i] * input[i];
    }

    return output;
}

double Neuron::backward(vector<double> w, vector<double> d, double outVal) {
    double s = 0;
    for (decltype(d.size()) i = 0; i < d.size(); ++i) {
        s += w[i] * d[i];
    }

    delta = (1 - outVal) * (1 - outVal) * s;
    return delta;
}
