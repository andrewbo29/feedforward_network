#include <vector>

using namespace std;

#ifndef FEEDFORWARD_NETWORK_NEURON_H
#define FEEDFORWARD_NETWORK_NEURON_H


class Neuron {
public:
    Neuron();
    double forward(vector<double> &input);
    void backward(vector<double> &w, vector<double> &d, double dActive);
    vector<double> &get_weights() { return weights; }
    double get_delta() { return delta; }
    void update_weights(double learning_rate);

private:
    vector<double> x;
    vector<double> weights;
    double delta;
};


#endif //FEEDFORWARD_NETWORK_NEURON_H
