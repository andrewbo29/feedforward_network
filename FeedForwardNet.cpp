#include "FeedForwardNet.h"

void FeedForwardNet::addFullyConnectedLayer(int size) {
    FullyConnectedLayer fc(size);
    fcs.push_back(fc);
}

vector<double> FeedForwardNet::forwardPass(vector<double> input) {
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





