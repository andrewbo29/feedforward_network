#include <vector>
#include <string>
#include <memory>
#include "Neuron.h"
#include "Activation.h"

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
    vector<double> get_grad();
    void set_weights(vector<vector<double>> &new_weights);

private:
    vector<Neuron> neurons;
    vector<shared_ptr<Activation>> activations;
};


#endif //FEEDFORWARD_NETWORK_FULLYCONNECTEDLAYER_H
