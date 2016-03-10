#include <vector>

using namespace std;

#ifndef FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H
#define FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H


class EuclideanLoss {
public:
    EuclideanLoss() = default;
    double loss(double x, int y);
    double backward(vector<double> w, vector<double> d, double dActive);
};


#endif //FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H
