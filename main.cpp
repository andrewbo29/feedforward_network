#include <iostream>
#include "FeedForwardNet.h"

using namespace std;

int main() {
    vector<double> data = {1, 2, 3};

    FeedForwardNet net;
    net.addFullyConnectedLayer(3);
    net.addFullyConnectedLayer(2);

    vector<double> res = net.forwardPass(data);

    for (auto &r : res) {
        cout << r << "\t";
    }
    cout << endl;

    return 0;
}