//
// Created by boyarov on 12.05.16.
//

#ifndef FEEDFORWARD_NETWORK_STEPDOWNPOLICY_H
#define FEEDFORWARD_NETWORK_STEPDOWNPOLICY_H


#include "LearningRatePolicy.h"

class StepDownPolicy : public LearningRatePolicy
{
public:
    StepDownPolicy() = default;
    StepDownPolicy(double lr, int en, std::vector<double> params) : LearningRatePolicy(lr, en), step_size(params[0]),
                                                                    gamma(params[1]) { }
    double change_learning_rate();

private:
    int step_size;
    double gamma;
};


#endif //FEEDFORWARD_NETWORK_STEPDOWNPOLICY_H
