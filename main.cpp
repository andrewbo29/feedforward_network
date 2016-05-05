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

    int epoch_num = 10;
    double learning_rate = 0.01;
    int batch_size = 1;

    net.train(dataTrain, labelsTrain, epoch_num, learning_rate, batch_size);

    double res = net.forwardPass(dataTrain[0]);

    cout << res << endl;

    return 0;
}

