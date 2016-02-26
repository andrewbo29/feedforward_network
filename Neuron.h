#include <vector>

using namespace std;

#ifndef FEEDFORWARD_NETWORK_NEURON_H
#define FEEDFORWARD_NETWORK_NEURON_H


class Neuron {
public:
    Neuron() = default;
    double forward(vector<double> input);
    double backward(vector<double> w, vector<double> d, double outVal);

private:
    vector<double> values;
    vector<double> weights;
    double delta;
};


#endif //FEEDFORWARD_NETWORK_NEURON_H
