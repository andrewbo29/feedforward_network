
#ifndef FEEDFORWARD_NETWORK_FEEDFORWARDNET_H
#define FEEDFORWARD_NETWORK_FEEDFORWARDNET_H


#include "InputLayer.h"
#include "FullyConnectedLayer.h"
#include "EuclideanLoss.h"

class FeedForwardNet {
public:
    FeedForwardNet() = default;
    FeedForwardNet(vector<double> m) : inputLayer(m) {};
    void addFullyConnectedLayer(int size, string activation_type);
    void addLossLayer(string losslayer_type);
    double forwardPass(vector<double> &input);
    double backwardPass(double input, int label);
    void train(vector<vector<double>> &data, vector<int> &labels, int iter_number, double learning_rate);
    void update_weights(double learning_rate);
    vector<vector<vector<double>>> get_weights();
    vector<double> get_grad();
    void set_weights(vector<vector<vector<double>>> &new_weights);

private:
    InputLayer inputLayer;
    vector<FullyConnectedLayer> fcs;
    EuclideanLoss lossLayer;
};


#endif //FEEDFORWARD_NETWORK_FEEDFORWARDNET_H
