#include "Tanh.h"

double Tanh::forward(double x) {
    s = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    return s;
}

double Tanh::backward() {
    return 1 - s * s;
}



