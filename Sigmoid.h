#include <vector>
#include <cmath>

using namespace std;

#ifndef FEEDFORWARD_NETWORK_SIGMOID_H
#define FEEDFORWARD_NETWORK_SIGMOID_H


class Sigmoid {
public:
    Sigmoid() = default;
    double forward(double x);
    double backward(double val);
};


#endif //FEEDFORWARD_NETWORK_SIGMOID_H
