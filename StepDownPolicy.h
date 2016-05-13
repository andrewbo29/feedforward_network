//
// Created by boyarov on 12.05.16.
//

#ifndef FEEDFORWARD_NETWORK_STEPDOWNPOLICY_H
#define FEEDFORWARD_NETWORK_STEPDOWNPOLICY_H


#include "LearningRatePolicy.h"

class StepDownPolicy : public LearningRatePolicy {
public:
    StepDownPolicy() = default;
    StepDownPolicy(double lr, int en, std::vector<double> params);
    double change_learning_rate(int cur_epoch) override;

private:
    double gamma;
    int epoch_step;
    int previous_change_epoch;
};


#endif //FEEDFORWARD_NETWORK_STEPDOWNPOLICY_H
