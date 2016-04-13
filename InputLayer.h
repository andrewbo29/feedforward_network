#include <vector>

using namespace std;

#ifndef FEEDFORWARD_NETWORK_INPUTLAYER_H
#define FEEDFORWARD_NETWORK_INPUTLAYER_H


class InputLayer {
public:
    InputLayer() = default;
    InputLayer(vector<double> m) : mean(m) {};
    void addData(vector<double> &input);
    vector<double> forward();

private:
    vector<double> data;
    vector<double> mean;
};


#endif //FEEDFORWARD_NETWORK_INPUTLAYER_H
