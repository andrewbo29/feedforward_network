#include <vector>

using namespace std;

#ifndef FEEDFORWARD_NETWORK_INPUTLAYER_H
#define FEEDFORWARD_NETWORK_INPUTLAYER_H


class InputLayer {
public:
    InputLayer() = default;
    vector<double> forward(vector<double> input);
};


#endif //FEEDFORWARD_NETWORK_INPUTLAYER_H
