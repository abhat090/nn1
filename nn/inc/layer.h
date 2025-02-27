// nn1::layer by Arya Bhattacharjee

/*
 * Subclass to network storing per-layer data.
 *
 * Contains the operations for layer forwarding and
 * setting intial weights and biases
 */

#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>
#include <util.h>

class NeuronLayer {
public:
    Eigen::MatrixXd weights_;
    Eigen::VectorXd biases_;
    Eigen::VectorXd z_;
    Eigen::VectorXd results_;
    int depth_;
    int size_;

    NeuronLayer();
    void init(int,int);
    void populate(RanD&, double);
    void forward(const Eigen::VectorXd& v, bool);

    int depth() const { return this->depth_; }
    int size() const { return this->size_; }

    friend std::ostream& operator<<(std::ostream& os, const NeuronLayer& nl) {
        os << nl.results_.transpose();
        return os;
    }
};

#endif //LAYER_H
