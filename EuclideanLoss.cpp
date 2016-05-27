#include "EuclideanLoss.h"

void EuclideanLoss::loss(vector<double> x, vector<int> y) {
    this->x = x;
    this->y = y;
    return;
//    return (x - y) * (x - y);
}

double EuclideanLoss::backward() {
    double sum = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += 2 * (x[i] - y[i]);
    }

    return sum;
}

double EuclideanLoss::compute_loss(vector<double> x, vector<int> y) {
    double sum = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sum;
}





