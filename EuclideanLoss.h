#include <vector>
#include <string>
#include "Sigmoid.h"

using namespace std;

#ifndef FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H
#define FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H


class EuclideanLoss {
public:
    EuclideanLoss() = default;
    double loss(double x, int y);
    double backward();

private:
    double x;
    int y;
};


#endif //FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H
