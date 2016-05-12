#ifndef FEEDFORWARD_NETWORK_LEARNINGRATEPOLICY_H
#define FEEDFORWARD_NETWORK_LEARNINGRATEPOLICY_H

#include <vector>

class LearningRatePolicy {
public:
    LearningRatePolicy(double lr, int en) : init_lr(lr), epoch_num(en) { }
    virtual double change_learning_rate() { return 0; };
    virtual ~LearningRatePolicy() = default;
protected:
    double init_lr;
    int epoch_num;
};

#endif //FEEDFORWARD_NETWORK_LEARNINGRATEPOLICY_H
