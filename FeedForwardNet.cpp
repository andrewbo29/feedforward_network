#include <stdexcept>
#include <iostream>
#include "FeedForwardNet.h"

void FeedForwardNet::addFullyConnectedLayer(int size, string activation_type) {
    FullyConnectedLayer fc(size, activation_type);
    fcs.push_back(fc);
}

double FeedForwardNet::forwardPass(vector<double> &input) {
    inputLayer.addData(input);

    vector<double> layerInput = inputLayer.forward();
    vector<double> layerOutput;

    for (auto &fc : fcs) {
        layerOutput = fc.forward(layerInput);
        layerInput = layerOutput;
        layerInput.insert(layerInput.begin(), 1.);
    }

    return layerOutput[0];
}

void FeedForwardNet::addLossLayer(string losslayer_type) {
    if (losslayer_type == "Euclidean") {
        lossLayer = EuclideanLoss();
    } else {
        throw runtime_error("No such activation function: " + losslayer_type);
    }
}

void FeedForwardNet::backwardPass(double input, int label) {
    cout << "Loss: " << lossLayer.loss(input, label) << endl;
    double loss_back = lossLayer.backward();
    vector<double> deltas = {loss_back};
    vector<vector<double>> weights;

    for (size_t i = fcs.size() - 1; i >= 0; --i) {
        fcs[i].backward(weights, deltas);
        deltas = fcs[i].get_deltas();
        weights = fcs[i].get_weights();
    }
}

void FeedForwardNet::train(vector<vector<double>> &data, vector<int> &labels, int iter_number) {
    int num = 0;
    while (num < iter_number) {
        for (size_t i = 0; i < data.size(); ++i) {
            vector<double> input = data[i];
            int label = labels[i];
            double out = forwardPass(input);
            backwardPass(out, label);
        }
        ++num;
    }
}











