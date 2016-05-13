#include <iostream>
#include "StepDownPolicy.h"

double StepDownPolicy::change_learning_rate(int cur_epoch) {
    if (cur_epoch % epoch_step == 0 && cur_epoch != previous_change_epoch) {
        init_lr *= gamma;
        previous_change_epoch = cur_epoch;
    }
    return init_lr;
}

StepDownPolicy::StepDownPolicy(double lr, int en, std::vector<double> params) : LearningRatePolicy(lr, en) {
    int step_size = static_cast<int>(params[0]);
    epoch_step = epoch_num * step_size / 100;
    gamma = params[1];
    previous_change_epoch = -1;
}



