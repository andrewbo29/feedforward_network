#include <vector>
#include <cmath>
#include "Activation.h"

using namespace std;

#ifndef FEEDFORWARD_NETWORK_SIGMOID_H
#define FEEDFORWARD_NETWORK_SIGMOID_H


class Sigmoid : public Activation {
public:
    Sigmoid() = default;
    double forward(double x) override;
    double backward() override;
};


#endif //FEEDFORWARD_NETWORK_SIGMOID_H
