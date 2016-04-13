#include <vector>
#include <string>
#include "Neuron.h"
#include "Tanh.h"

using namespace std;

#ifndef FEEDFORWARD_NETWORK_FULLYCONNECTEDLAYER_H
#define FEEDFORWARD_NETWORK_FULLYCONNECTEDLAYER_H


class FullyConnectedLayer {
public:
    FullyConnectedLayer(int size, string activation_type);
    vector<double> forward(vector<double> &input);
    void backward(vector<vector<double>> &weights, vector<double> &deltas);
    vector<vector<double>> get_weights();
    vector<double> get_deltas();
    void update_weights(double learning_rate);

private:
    vector<Neuron> neurons;
    vector<Tanh> activations;
};


#endif //FEEDFORWARD_NETWORK_FULLYCONNECTEDLAYER_H
