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

double FeedForwardNet::backwardPass(double input, int label) {
    double loss = lossLayer.loss(input, label);
    double loss_back = lossLayer.backward();
    vector<double> deltas = {loss_back};
    vector<vector<double>> weights;

    for (int i = (int)fcs.size() - 1; i >= 0; --i) {
        fcs[i].backward(weights, deltas);
        deltas = fcs[i].get_deltas();
        weights = fcs[i].get_weights();
    }

    return loss;
}

void FeedForwardNet::train(vector<vector<double>> &data, vector<int> &labels, int iter_number, double learning_rate) {
    int num = 0;
    while (num < iter_number) {
        cout << "Train iteration: " << num << endl;
        double sum_loss = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            vector<double> input = data[i];
            int label = labels[i];
            double out = forwardPass(input);
            sum_loss += backwardPass(out, label);
            update_weights(learning_rate);
        }
        cout << "Mean epoch loss: " << sum_loss / data.size() << endl;
        ++num;
    }
}

void FeedForwardNet::update_weights(double learning_rate) {
    for (size_t i = 0; i < fcs.size(); ++i) {
//        cout << "Layer " << i << ":" << endl;
        fcs[i].update_weights(learning_rate);
    }
}

vector<vector<vector<double>>> FeedForwardNet::get_weights() {
    vector<vector<vector<double>>> all_weights;
    for (auto &fc : fcs) {
        all_weights.push_back(fc.get_weights());
    }
    return all_weights;
}

void FeedForwardNet::set_weights(vector<vector<vector<double>>> &new_weights) {
    for (size_t i = 0; i < fcs.size(); ++i) {
        fcs[i].set_weights(new_weights[i]);
    }
}

vector<double> FeedForwardNet::get_grad() {
    vector<double> grad;
    for (auto &fc : fcs) {
        for (auto &g : fc.get_grad()) {
            grad.push_back(g);
        }
    }
    return grad;
}























