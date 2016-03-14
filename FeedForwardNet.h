
#ifndef FEEDFORWARD_NETWORK_FEEDFORWARDNET_H
#define FEEDFORWARD_NETWORK_FEEDFORWARDNET_H


#include "InputLayer.h"
#include "FullyConnectedLayer.h"
#include "EuclideanLoss.h"

class FeedForwardNet {
public:
    FeedForwardNet() = default;
    void addFullyConnectedLayer(int size, string activation_type);
    void addLossLayer(string losslayer_type);
    vector<double> forwardPass(vector<double> &input);
    void backwardPass(vector<double> &input, int label);

private:
    InputLayer inputLayer;
    vector<FullyConnectedLayer> fcs;
    EuclideanLoss lossLayer;
};


#endif //FEEDFORWARD_NETWORK_FEEDFORWARDNET_H
