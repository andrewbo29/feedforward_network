
#ifndef FEEDFORWARD_NETWORK_FEEDFORWARDNET_H
#define FEEDFORWARD_NETWORK_FEEDFORWARDNET_H


#include "InputLayer.h"
#include "FullyConnectedLayer.h"
#include "EuclideanLoss.h"
#include "LearningRatePolicy.h"

class FeedForwardNet {
public:
    FeedForwardNet() = default;
    FeedForwardNet(vector<double> m) : inputLayer(m) {}
    void addFullyConnectedLayer(int size, string activation_type);
    void addLossLayer(string losslayer_type);
    double forwardPass(vector<double> &input);
    void backwardPass(double input, int label);
    vector<double> train(vector<vector<double>> &data, vector<int> &labels, int iter_number, double learning_rate,
                         LearningRatePolicy *lr_policy, int batch_size);
    void update_weights(double learning_rate);
    vector<vector<vector<double>>> get_weights();
    vector<double> get_grad();
    void set_weights(vector<vector<vector<double>>> &new_weights);
    double get_loss(double x, int y);
    vector<vector<size_t>> get_batches(vector<vector<double>> &data, int batch_size);

private:
    InputLayer inputLayer;
    vector<FullyConnectedLayer> fcs;
    EuclideanLoss lossLayer;
};


#endif //FEEDFORWARD_NETWORK_FEEDFORWARDNET_H
