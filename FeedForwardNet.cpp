#include <stdexcept>
#include "FeedForwardNet.h"

void FeedForwardNet::addFullyConnectedLayer(int size, string activation_type) {
    FullyConnectedLayer fc(size, activation_type);
    fcs.push_back(fc);
}

vector<double> FeedForwardNet::forwardPass(vector<double> &input) {
    inputLayer.addData(input);

    vector<double> layerInput = inputLayer.forward();
    vector<double> layerOutput;

    for (auto &fc : fcs) {
        layerOutput = fc.forward(layerInput);
        layerInput = layerOutput;
        layerInput.insert(layerInput.begin(), 1.);
    }

    return layerOutput;
}

void FeedForwardNet::addLossLayer(string losslayer_type) {
    if (losslayer_type == "Euclidean") {
        lossLayer = EuclideanLoss();
    } else {
        throw runtime_error("No such activation function: " + losslayer_type);
    }
}


void FeedForwardNet::backwardPass() {
    for (auto &fc : fcs) {
        fc.backward()
    }
}







