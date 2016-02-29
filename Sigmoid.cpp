#include "Sigmoid.h"

double Sigmoid::forward(double x) {
    return 1 / (1 + exp(x));
}

double Sigmoid::backward(double val) {
    double s = forward(val);
    return s * (1 - s);
}



