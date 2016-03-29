#include <iostream>
#include "FeedForwardNet.h"
#include "imageProcessing.h"

using namespace std;

int main() {
//    vector<vector<double>> data = readImagesDir("/home/boyarov/Projects/cpp/data/mnist_data_0/");

    FeedForwardNet net;
    net.addFullyConnectedLayer(3, "Sigmoid");
    net.addFullyConnectedLayer(2, "Sigmoid");
    net.addFullyConnectedLayer(1, "Sigmoid");
    net.addLossLayer("Euclidean");

    string posDirNameTrain = "/media/datab/bases/mnist/train/0";
    string negDirNameTrain = "/media/datab/bases/mnist/train/1";

    cout << "Load train data" << endl;

    vector<vector<double>> dataTrain;
    vector<int> labelsTrain;

    readImagesData(posDirNameTrain, negDirNameTrain, dataTrain, labelsTrain);

    cout << "Train net" << endl;

    int iter_num = 100;
    double learning_rate = 0.01;

    net.train(dataTrain, labelsTrain, iter_num, learning_rate);

//    double res = net.forwardPass(dataTrain[0]);
//
//    cout << res << endl;

    return 0;
}