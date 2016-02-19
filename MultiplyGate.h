#include <vector>
using namespace std;

#ifndef FEEDFORWARD_NETWORK_MULTIPLYGATE_H
#define FEEDFORWARD_NETWORK_MULTIPLYGATE_H


class MultiplyGate {
public:
    MultiplyGate() = default;
    double forward(double a, double b);
    vector<double> backward(double da);

private:
    double x;
    double y;
};


#endif //FEEDFORWARD_NETWORK_MULTIPLYGATE_H
