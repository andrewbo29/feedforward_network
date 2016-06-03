#include <iostream>
#include <highgui.h>
#include <fstream>
#include "FeedForwardNet.h"
#include "imageProcessing.h"
#include "StepDownPolicy.h"

using namespace std;

void save_loss(string fname, vector<double> loss) {
    ofstream output(fname);

    for (auto const &l : loss) {
        output << l << endl;
    }

    return;
}


int main() {
    string posDirNameTrain = "/media/datab/bases/mnist/train/0";
    string negDirNameTrain = "/media/datab/bases/mnist/train/1";

//    string posDirNameTrain = "/home/boyarov/Projects/cpp/data/mnist_data_0";
//    string negDirNameTrain = "/home/boyarov/Projects/cpp/data/mnist_data_1";

    cout << "Load train data" << endl;

    vector<string> dirs_list = {posDirNameTrain, negDirNameTrain};

    vector<vector<double>> dataTrain;
    vector<vector<int>> labelsTrain;

    readImagesData(dirs_list, dataTrain, labelsTrain);

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

    net.addFullyConnectedLayer(4, "Tanh");
    net.addFullyConnectedLayer(3, "Tanh");
    net.addFullyConnectedLayer(2, "Tanh");
    net.addLossLayer("Euclidean");

    cout << "Train net" << endl;

    int epoch_num = 20;
    double learning_rate = 0.001;
    int batch_size = 256;

    StepDownPolicy step_down = StepDownPolicy(learning_rate, epoch_num, {20, 0.1});
    LearningRatePolicy *lr_policy = &step_down;

    vector<double> loss = net.train(dataTrain, labelsTrain, epoch_num, learning_rate, lr_policy, batch_size);

    string loss_fname = "/home/boyarov/Projects/cpp/feedforward_network/log_loss.txt";
    save_loss(loss_fname, loss);

//    double res = net.forwardPass(dataTrain[0]);
//
//    cout << res << endl;

    return 0;
}
