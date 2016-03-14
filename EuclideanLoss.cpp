#include "EuclideanLoss.h"

double EuclideanLoss::loss(double x, int y) {
    this->x = x;
    this->y = y;
    return (x - y) * (x - y);
}

double EuclideanLoss::backward() {
    return 2 * (x - y);
}



