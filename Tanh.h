#include <vector>
#include <cmath>

#ifndef FEEDFORWARD_NETWORK_TANH_H
#define FEEDFORWARD_NETWORK_TANH_H


class Tanh
{
public:
    Tanh() = default;
    double forward(double x);
    double backward();
private:
    double s;
};


#endif //FEEDFORWARD_NETWORK_TANH_H
