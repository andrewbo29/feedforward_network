#include <vector>
#include <string>
#include "Sigmoid.h"

using namespace std;

#ifndef FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H
#define FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H


class EuclideanLoss {
public:
    EuclideanLoss() = default;
    void loss(double x, int y);
    double backward();
    double compute_loss(double x, int y);

private:
    double x;
    int y;
};


#endif //FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H
