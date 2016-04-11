#include <iostream>
#include "FeedForwardNet.h"
#include "imageProcessing.h"

using namespace std;

int main() {
    FeedForwardNet net;
//    net.addFullyConnectedLayer(3, "Sigmoid");
//    net.addFullyConnectedLayer(2, "Sigmoid");
//    net.addFullyConnectedLayer(1, "Sigmoid");
    net.addFullyConnectedLayer(3, "Tanh");
    net.addFullyConnectedLayer(2, "Tanh");
    net.addFullyConnectedLayer(1, "Tanh");
    net.addLossLayer("Euclidean");

//    string posDirNameTrain = "/media/datab/bases/mnist/train/0";
//    string negDirNameTrain = "/media/datab/bases/mnist/train/1";

    string posDirNameTrain = "/home/boyarov/Projects/cpp/data/mnist_data_0";
    string negDirNameTrain = "/home/boyarov/Projects/cpp/data/mnist_data_1";

    cout << "Load train data" << endl;

    vector<vector<double>> dataTrain;
    vector<int> labelsTrain;

    readImagesData(posDirNameTrain, negDirNameTrain, dataTrain, labelsTrain);

    cout << "Train net" << endl;

    int iter_num = 3;
    double learning_rate = 0.01;

    net.train(dataTrain, labelsTrain, iter_num, learning_rate);

//    double res = net.forwardPass(dataTrain[0]);
//
//    cout << res << endl;

    return 0;
}