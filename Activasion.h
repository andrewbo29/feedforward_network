#ifndef FEEDFORWARD_NETWORK_ACTIVASION_H
#define FEEDFORWARD_NETWORK_ACTIVASION_H


class Activasion {
public:
    Activasion() = default;
    double forward(double x);
    double backward();
};


#endif //FEEDFORWARD_NETWORK_ACTIVASION_H
