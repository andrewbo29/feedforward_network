#include <stdexcept>
#include <iostream>
#include <chrono>
#include "FeedForwardNet.h"
#include <random>
#include <algorithm>

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
//        for (auto &l : layerOutput) {
//            std::cout << l << " " ;
//        }
//        std::cout << std::endl;
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

void FeedForwardNet::backwardPass(vector<double> input, vector<int> label) {
    lossLayer.loss(input, label);
//    double loss_back = lossLayer.backward();
//    vector<double> deltas = {loss_back};
    vector<double> deltas = lossLayer.backward();
    vector<vector<double>> weights;

    for (int i = (int)fcs.size() - 1; i >= 0; --i) {
        fcs[i].backward(weights, deltas);
        deltas = fcs[i].get_deltas();
        weights = fcs[i].get_weights();
    }

    return;
}

vector<double> FeedForwardNet::train(vector<vector<double>> &data, vector<vector<int>> &labels, int epoch_number,
                                     double learning_rate, LearningRatePolicy *lr_policy, int batch_size=1) {

    vector<double> loss;

    int num_epoch = 1;
    while (num_epoch <= epoch_number) {
        vector<vector<size_t>> batches_ind = get_batches(data, batch_size);
        int iter_num = 0;
        for (auto &batch_ind : batches_ind) {
            double sum_loss = 0;
            for (size_t i : batch_ind) {
                vector<double> input = data[i];
                vector<int> label = labels[i];
                vector<double> out = forwardPass(input);
                backwardPass(out, label);
                sum_loss += lossLayer.compute_loss(out, label);
            }
            double mean_loss = sum_loss / batch_ind.size();
            if (std::isnan(mean_loss)) {
//                std::cout << sum_loss << " " << batch_ind.size() << std::endl;
                return loss;
            }
            cout << "Epoch " << num_epoch << ", iteration " << iter_num << ", learning rate " << learning_rate << ", loss " << mean_loss << endl;
            loss.push_back(mean_loss);
            learning_rate = lr_policy->change_learning_rate(num_epoch);
            update_weights(learning_rate);
            ++iter_num;
        }
        ++num_epoch;
    }

    return loss;
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

double FeedForwardNet::get_loss(vector<double> x, vector<int> y) {
    return lossLayer.compute_loss(x, y);
}

vector<vector<size_t>> FeedForwardNet::get_batches(vector<vector<double>> &data, int batch_size) {
    vector<size_t> data_ind;
    for (size_t i = 0; i < data.size(); ++i) {
        data_ind.push_back(i);
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(data_ind.begin(), data_ind.end(), std::default_random_engine(seed));

    vector<vector<size_t>> batches_ind;

    size_t ind;
    for (ind = 0; ind < data_ind.size(); ind += batch_size) {
        vector<size_t> batch_ind;
        for (size_t j = ind; j < ind + batch_size && j < data_ind.size(); ++j) {
            batch_ind.push_back(data_ind[j]);
        }
        batches_ind.push_back(batch_ind);
    }

    ind -= batch_size;
    vector<size_t> batch_ind;
    while (ind < data_ind.size()) {
        batch_ind.push_back(data_ind[ind]);
        ++ind;
    }
    batches_ind.push_back(batch_ind);

    return batches_ind;
}



























