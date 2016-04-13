#include "InputLayer.h"

vector<double> InputLayer::forward() {
    return data;
}

void InputLayer::addData(vector<double> &input) {
    data = input;
    if (!mean.empty()) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] -= mean[i];
        }
    }
    data.insert(data.begin(), 1.);
}

