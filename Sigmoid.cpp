#include "Sigmoid.h"

double Sigmoid::forward(double x) {
    s = 1 / (1 + exp(x));
    return s;
}

double Sigmoid::backward() {
    return s * (1 - s);
}



