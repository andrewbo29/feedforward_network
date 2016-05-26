#include <iostream>
#include "Tanh.h"

double Tanh::forward(double x) {
    s = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
//    if (std::isnan(s)) {
//        std::cout << x << std::endl;
//    }
//    std::cout << "  " << x << " " << exp(x) << " " << s << std::endl;
    return s;
}

double Tanh::backward() {
    return 1 - s * s;
}



