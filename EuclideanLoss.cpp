#include <iostream>
#include "EuclideanLoss.h"

void EuclideanLoss::loss(vector<double> x, vector<int> y) {
    this->x = x;
    this->y = y;
    return;
//    return (x - y) * (x - y);
}

vector<double> EuclideanLoss::backward() {
    vector<double> deltas;

//    deltas.push_back(0);
//    for (size_t i = 0; i < x.size(); ++i) {
//        deltas[0] += (x[i] - y[i]);
//    }
//    deltas[0] = 2 * sqrt(deltas[0]);

    for (size_t i = 0; i < x.size(); ++i) {
        deltas.push_back(x[i] - y[i]);
    }

    return deltas;
}

double EuclideanLoss::compute_loss(vector<double> x, vector<int> y) {
    double sum = 0;
    for (size_t i = 0; i < x.size(); ++i) {
//        std::cout << x[i] << " " << y[i] << std::endl;
        sum += 0.5 * (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sum;
}





