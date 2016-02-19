#include "MultiplyGate.h"

double MultiplyGate::forward(double a, double b) {
    double z = a * b;
    x = a;
    y = b;
    return z;
}

vector<double> MultiplyGate::backward(double da) {
    double dx = x * da;
    double dy = y * da;
    return vector<double> {dx, dy};
}
