// nn1::network by Arya Bhattacharjee

/*
 * Primary class for controlling the Neural Network
 *
 * Configurable layers and depth per layer
 * Primarily uses Eigen linear algebra library for fast calculations
 *
 * Note to initialize neural network before attempting to forward
 */

#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <layer.h>

// Tuple containing weight gradient in matrix form, and bias gradient in vector form
typedef std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>> gradient;

class Network {
    RanD r = RanD(-1, 1);

    std::vector<NeuronLayer> net_layers_;

    int start_depth_;
    int layer_count_;
    int layer_depth_;
    int result_depth_;

    double learn_rate_;

    bool init_cond_ = false;

public:
    Network();

    void init(Eigen::VectorXd&, int);
    void apply(std::vector<Eigen::MatrixXd>&, std::vector<Eigen::VectorXd>&, int);

    int forward(const Eigen::VectorXd&);
    gradient cycle(const Eigen::VectorXd&, const Eigen::VectorXd&);
    gradient backprop(const Eigen::VectorXd&, const Eigen::VectorXd&);

    void setLayerCount(const int l) { if(!init_cond_) this->layer_count_=l; }
    void setLayerDepth(const int d) { if(!init_cond_) this->layer_depth_=d; }
    void setResultDepth(const int r) { if(!init_cond_) this->result_depth_=r; }
    void setLearnRate(const double lr) { if(!init_cond_) this->learn_rate_=lr; }

    std::vector<NeuronLayer>& layers() { return this->net_layers_; }

    friend std::ostream& operator<<(std::ostream& os, const Network& net) {
        os << net.net_layers_.back();
        return os;
    }

private:
    void populate(Eigen::VectorXd&);
};

#endif //NETWORK_H
