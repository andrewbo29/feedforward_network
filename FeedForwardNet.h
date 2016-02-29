
#ifndef FEEDFORWARD_NETWORK_FEEDFORWARDNET_H
#define FEEDFORWARD_NETWORK_FEEDFORWARDNET_H


#include "InputLayer.h"
#include "FullyConnectedLayer.h"

class FeedForwardNet {
public:
    FeedForwardNet() = default;
    void addLayer(int size);
    double forwardPass();

private:
    InputLayer inputLayer;
    vector<FullyConnectedLayer> fcs;
};


#endif //FEEDFORWARD_NETWORK_FEEDFORWARDNET_H
