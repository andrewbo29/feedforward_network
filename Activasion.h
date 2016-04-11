#ifndef FEEDFORWARD_NETWORK_ACTIVASION_H
#define FEEDFORWARD_NETWORK_ACTIVASION_H


class Activasion {
public:
    Activasion() = default;
    virtual double forward(double x);
    virtual double backward();
private:
    double s;
};


#endif //FEEDFORWARD_NETWORK_ACTIVASION_H
