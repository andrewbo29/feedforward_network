#include <vector>
#include <cmath>
#include "Activasion.h"

using namespace std;

#ifndef FEEDFORWARD_NETWORK_SIGMOID_H
#define FEEDFORWARD_NETWORK_SIGMOID_H


class Sigmoid
{
public:
    Sigmoid() = default;
    double forward(double x);
    double backward();
private:
    double s;
};


#endif //FEEDFORWARD_NETWORK_SIGMOID_H
