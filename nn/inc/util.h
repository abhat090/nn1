// nn1::util by Arya Bhattacharjee

/*
 * Header file that contains some basic non standard
 * math functions for use in the nn algorithm
 */

#ifndef UTIL_H
#define UTIL_H

#include <random>
#include <Eigen/Dense>

/*
 * Simple wrapper class for c++ random library
 * Used for initialization over Eigen's random for
 * better control over starting weights
 */
class RanD {
    std::random_device r;
    std::default_random_engine rand_;
    std::uniform_real_distribution<double> dist_;

public:
    RanD(double l, double h) {
        rand_ = std::default_random_engine(r());
        dist_ = std::uniform_real_distribution<double>(l, h);
    }

    double gen() {
        return dist_(rand_);
    }
};

// Takes a digit and converts it to a vector index
static Eigen::VectorXd yVec(int expected, int size, double factor) {
    Eigen::VectorXd v(size);
    v[expected] = factor;
    return v;
}

// Efficient ReLU implementation
static double ReLU(double i) {
    return i * (i > 0);
}

// Standard sigmoid expression used for final layer
static double Sigmoid(double i) {
    return 1.0/(1.0+exp(-i));
}

// Applies ReLU derivative to a whole vector
static Eigen::VectorXd ReLU_P(const Eigen::VectorXd &z) {
    Eigen::VectorXd r(z.size());
    for(int i = 0; i < z.size(); i++) {
        r[i] = static_cast<double>(z[i] > 0);
    }
    return r;
}

// Applies sigmoid derivative to a whole vector
static Eigen::VectorXd Sigmoid_P(const Eigen::VectorXd &z) {
    Eigen::VectorXd r(z.size());
    for(int i = 0; i < z.size(); i++) {
        r[i] = Sigmoid(z[i])*(1-Sigmoid(z[i]));
    }
    return r;
}

// Derivative of network cost function
static Eigen::VectorXd del_cost(const Eigen::VectorXd &a_L, const Eigen::VectorXd &y) {
    return 2 * (a_L - y);
}

#endif //UTIL_H
