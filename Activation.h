//
// Created by boyarov on 21.04.16.
//

#ifndef FEEDFORWARD_NETWORK_ACTIVATION_H
#define FEEDFORWARD_NETWORK_ACTIVATION_H

class Activation {
public:
    Activation() = default;
    virtual double forward(double x) { };
    virtual double backward() { };
    virtual ~Activation() = default;
protected:
    double s;
};

#endif //FEEDFORWARD_NETWORK_ACTIVATION_H
