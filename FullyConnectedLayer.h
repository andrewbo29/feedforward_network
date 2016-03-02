#include <vector>
#include <string>
#include "Neuron.h"
#include "Sigmoid.h"

using namespace std;

#ifndef FEEDFORWARD_NETWORK_FULLYCONNECTEDLAYER_H
#define FEEDFORWARD_NETWORK_FULLYCONNECTEDLAYER_H


class FullyConnectedLayer {
public:
    FullyConnectedLayer(int size, string activation_type);
    vector<double> forward(vector<double> &input);

private:
    vector<Neuron> neurons;
    vector<Sigmoid> activations;
};


#endif //FEEDFORWARD_NETWORK_FULLYCONNECTEDLAYER_H
