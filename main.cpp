#include <iostream>
#include "FeedForwardNet.h"
#include "imageProcessing.h"

using namespace std;

int main() {
    vector<vector<double>> data = readImagesDir("/home/boyarov/Projects/cpp/data/mnist_data_0/");

    FeedForwardNet net;
    net.addFullyConnectedLayer(3, "Sigmoid");
    net.addFullyConnectedLayer(2, "Sigmoid");
    net.addFullyConnectedLayer(1, "Sigmoid");

    vector<double> res = net.forwardPass(data[0]);

    for (auto &r : res) {
        cout << r << "\t";
    }
    cout << endl;

    return 0;
}