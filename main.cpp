#include <iostream>
#include <highgui.h>
#include "FeedForwardNet.h"
#include "imageProcessing.h"

using namespace std;

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

    cout << "Train net" << endl;

    int iter_num = 10;
    double learning_rate = 0.01;

    net.train(dataTrain, labelsTrain, iter_num, learning_rate);

    double res = net.forwardPass(dataTrain[0]);

    cout << res << endl;

    return 0;
}


void gradient_check(FeedForwardNet &net, vector<double> &data_elem, int label) {
    double res = net.forwardPass(data_elem);
    vector<vector<vector<double>>> weights = net.get_weights();

    FeedForwardNet net_1 = net;

    net.backwardPass(res, label);
    vector<double> analytic_grad = net.get_grad();

    double epsilon = 0.0001;
//    vector<vector<vector<double>>> weights_plus;
//    vector<vector<vector<double>>> weights_minus;
//    for (auto &fc_w : weights) {
//        vector<vector<double>> weights_plus_1;
//        vector<vector<double>> weights_minus_1;
//        for (auto &n_w : fc_w) {
//            vector<double> weights_plus_2;
//            vector<double> weights_minus_2;
//            for (auto &w : n_w) {
//                weights_plus_2.push_back(w + epsilon);
//                weights_minus_2.push_back(w - epsilon);
//            }
//            weights_plus_1.push_back(weights_plus_2);
//            weights_minus_1.push_back(weights_minus_2);
//        }
//        weights_plus.push_back(weights_plus_1);
//        weights_minus.push_back(weights_minus_1);
//    }
}