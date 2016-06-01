#include <vector>
#include <string>
#include "Sigmoid.h"

using namespace std;

#ifndef FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H
#define FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H


class EuclideanLoss {
public:
    EuclideanLoss() = default;
    void loss(vector<double> x, vector<int> y);
    vector<double> backward();
    double compute_loss(vector<double> x, vector<int> y);

private:
    vector<double> x;
    vector<int> y;
};


#endif //FEEDFORWARD_NETWORK_EUCLIDEANLOSS_H
