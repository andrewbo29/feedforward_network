#include <gtest/gtest.h>
#include "FeedForwardNet.h"
#include "imageProcessing.h"


double gradient_check(FeedForwardNet &net, vector<double> &data_elem, int label) {
    double res = net.forwardPass(data_elem);
    vector<vector<vector<double>>> weights = net.get_weights();

    FeedForwardNet net_1 = net;
    FeedForwardNet net_2 = net;

    net.backwardPass(res, label);
    vector<double> analytic_grad = net.get_grad();

    double epsilon = 0.0001;

    vector<vector<vector<double>>> weights_plus;
    vector<vector<vector<double>>> weights_minus;

    vector<double> numerical_grad;
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t z = 0; z < weights[i][j].size(); ++z) {
                weights_plus = weights;
                weights_minus = weights;

                weights_minus[i][j][z] -= epsilon;
                weights_plus[i][j][z] += epsilon;
                net_1.set_weights(weights_minus);
                net_2.set_weights(weights_plus);

                double minus_res = net_1.forwardPass(data_elem);
                double minus_loss = net_1.get_loss(minus_res, label);
                double plus_res = net_2.forwardPass(data_elem);
                double plus_loss = net_2.get_loss(plus_res, label);

                numerical_grad.push_back((plus_loss - minus_loss) / epsilon);
            }
        }
    }

    double sum = 0;
    for (size_t i = 0; i < analytic_grad.size(); ++i) {
        sum += (analytic_grad[i] - numerical_grad[i]) * (analytic_grad[i] - numerical_grad[i]);
    }

    return sqrt(sum) / analytic_grad.size();
}

TEST(test, gradient_check_test) {
    string posDirNameTrain = "/home/boyarov/Projects/cpp/data/mnist_data_0";
    string negDirNameTrain = "/home/boyarov/Projects/cpp/data/mnist_data_1";

    vector<vector<double>> dataTrain;
    vector<int> labelsTrain;

    readImagesData(posDirNameTrain, negDirNameTrain, dataTrain, labelsTrain);

    FeedForwardNet net = FeedForwardNet();

    net.addFullyConnectedLayer(3, "Tanh");
    net.addFullyConnectedLayer(2, "Tanh");
    net.addFullyConnectedLayer(1, "Tanh");
    net.addLossLayer("Euclidean");

    int ind = static_cast<int>(rand() % labelsTrain.size());
    double check_res = gradient_check(net, dataTrain[ind], labelsTrain[ind]);

    ASSERT_LE(check_res, 0.001);
}

