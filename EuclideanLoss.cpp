#include "EuclideanLoss.h"

void EuclideanLoss::loss(double x, int y) {
    this->x = x;
    this->y = y;
    return;
//    return (x - y) * (x - y);
}

double EuclideanLoss::backward() {
    return 2 * (x - y);
}

double EuclideanLoss::compute_loss(double x, int y) {
    return (x - y) * (x - y);
}





