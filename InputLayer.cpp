#include "InputLayer.h"

vector<double> InputLayer::forward() {
    return data;
}


void InputLayer::addData(vector<double> input) {
    data = input;
    data.insert(data.begin(), 1.);
}

