#include <iostream>
#include <highgui.h>
#include "FeedForwardNet.h"
#include "imageProcessing.h"

using namespace std;

void gradient_check(FeedForwardNet &net, vector<double> &data_elem, int label) {
    double res = net.forwardPass(data_elem);
    vector<vector<vector<double>>> weights = net.get_weights();

    FeedForwardNet net_1 = net;
    FeedForwardNet net_2 = net;

    net.backwardPass(res, label);
    vector<double> analytic_grad = net.get_grad();

    double epsilon = 0.0001;

    vector<vector<vector<double>>> weights_plus = weights;
    vector<vector<vector<double>>> weights_minus = weights;

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
//                cout << (plus_loss - minus_loss) / epsilon << endl;
            }
        }
    }

    double sum = 0;
    for (size_t i = 0; i < analytic_grad.size(); ++i) {
        sum += (analytic_grad[i] - numerical_grad[i]) * (analytic_grad[i] - numerical_grad[i]);
    }

    cout << "Gradient check error: " << sqrt(sum) / analytic_grad.size() << endl;
}

int main() {
//    string posDirNameTrain = "/media/datab/bases/mnist/train/0";
//    string negDirNameTrain = "/media/datab/bases/mnist/train/1";

    string posDirNameTrain = "/home/boyarov/Projects/cpp/data/mnist_data_0";
    string negDirNameTrain = "/home/boyarov/Projects/cpp/data/mnist_data_1";

    cout << "Load train data" << endl;

    vector<vector<double>> dataTrain;
    vector<int> labelsTrain;

    readImagesData(posDirNameTrain, negDirNameTrain, dataTrain, labelsTrain);

    vector<string> dirs_list = {posDirNameTrain, negDirNameTrain};

    bool is_compute_mean = false;
    string mean_image_filename = "/home/boyarov/Projects/cpp/data/mean_image.png";

    if (is_compute_mean) {
        cv::imwrite(mean_image_filename, compute_mean_image(dirs_list));
    }
    vector<double> mean_image = readImage(mean_image_filename);

    FeedForwardNet net = FeedForwardNet(mean_image);

//    net.addFullyConnectedLayer(3, "Sigmoid");
//    net.addFullyConnectedLayer(2, "Sigmoid");
//    net.addFullyConnectedLayer(1, "Sigmoid");

    net.addFullyConnectedLayer(3, "Tanh");
    net.addFullyConnectedLayer(2, "Tanh");
    net.addFullyConnectedLayer(1, "Tanh");
    net.addLossLayer("Euclidean");

//    cout << "Train net" << endl;
//
//    int iter_num = 10;
//    double learning_rate = 0.01;
//
//    net.train(dataTrain, labelsTrain, iter_num, learning_rate);
//
//    double res = net.forwardPass(dataTrain[0]);
//
//    cout << res << endl;

    int ind = static_cast<int>(rand() % labelsTrain.size());
    gradient_check(net, dataTrain[ind], labelsTrain[ind]);

    return 0;
}

