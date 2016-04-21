#include <vector>
#include <cmath>
#include "Activation.h"

#ifndef FEEDFORWARD_NETWORK_TANH_H
#define FEEDFORWARD_NETWORK_TANH_H


class Tanh : public Activation {
public:
    Tanh() = default;
    double forward(double x) override;
    double backward() override;
};


#endif //FEEDFORWARD_NETWORK_TANH_H
