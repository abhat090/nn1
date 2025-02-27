// nn1::networkcpp by Arya Bhattacharjee

#include <iostream>
#include <network.h>

using namespace std;

Network::Network() {
    this->start_depth_ = -1;
    this->learn_rate_ = 1.0;
    this->layer_count_
    = this->layer_depth_
    = this->result_depth_ = 1;
}

void Network::init(Eigen::VectorXd &input, int size) {
    if (this->init_cond_) return;

    this->start_depth_ = size;

    for(int i = 0; i < layer_count_ + 1; i++) {
        this->net_layers_.emplace_back();
    }

    this->populate(input);
    this->init_cond_ = true;
}

void Network::apply(std::vector<Eigen::MatrixXd> &del_w, std::vector<Eigen::VectorXd> &del_b, int batch_size) {
    int i = 0;
    for (NeuronLayer& nl : this->net_layers_) {
        nl.weights_ -= (this->learn_rate_) * del_w[i];
        nl.biases_ -= (this->learn_rate_) * del_b[i];
        i++;
    }
}


gradient Network::cycle(const Eigen::VectorXd &image, const Eigen::VectorXd &label) {
    this->forward(image);
    return this->backprop(label, image);
}

int Network::forward(const Eigen::VectorXd &input) {
    if (!this->init_cond_) return -1;

    this->net_layers_[0].forward(input, false);

    for (int i = 1; i < this->net_layers_.size() - 1; i++) {
        this->net_layers_[i].forward(this->net_layers_[i-1].results_, true);
    }

    this->net_layers_.back().forward(this->net_layers_[this->net_layers_.size()-2].results_, false);

    int index;
    this->net_layers_.back().results_.maxCoeff(&index);

    return index;
}

/*
 * Backpropagation algorithm
 *
 * Primary workhorse of nn, calculating gradient descent
 * of the network's weights and biases, in order to lower the cost function
 */
gradient Network::backprop(const Eigen::VectorXd &y, const Eigen::VectorXd &image) {
    vector<Eigen::MatrixXd> del_w; // weight gradient
    vector<Eigen::VectorXd> del_b; // bias gradient

    // Outer most chain rule; outer derivative of cost function
    Eigen::VectorXd delta = del_cost(
        net_layers_[layer_count_].results_, y).asDiagonal() * Sigmoid_P(net_layers_[layer_count_].z_
            );

    // Partial derivative of weight ends up being results of previous layer * Cost derivative
    Eigen::MatrixXd delta_w = delta * net_layers_[layer_count_-1].results_.transpose();

    del_b.push_back(delta);
    del_w.push_back(delta_w);

    // Calculates the hidden layer derivatives backwards using forward layers' weights
    auto r_itr = ++this->net_layers_.rbegin();
    while (r_itr != this->net_layers_.rend() - 1) {
        delta = ((r_itr - 1)->weights_.transpose() * delta).asDiagonal() * ReLU_P(r_itr->z_);
        delta_w = delta * (r_itr + 1)->results_.transpose();

        del_b.push_back(delta);
        del_w.push_back(delta_w);
        ++r_itr;
    }

    // First hidden layer requires seperate math to obtain image activation
    delta = ((r_itr - 1)->weights_.transpose() * delta).asDiagonal() * Sigmoid_P(r_itr->z_);
    delta_w = delta * image.transpose();

    del_b.push_back(delta);
    del_w.push_back(delta_w);

    // return gradient as tuple of vectors of respective matrix/vector
    return gradient(del_w, del_b);
}


void Network::populate(Eigen::VectorXd &input) {
    net_layers_[0].init(layer_depth_, start_depth_);
    net_layers_[0].populate(this->r, sqrt(2.00/start_depth_));
    net_layers_[0].forward(input, false);

    for (int i = 1; i < layer_count_; i++) {
        net_layers_[i].init(layer_depth_, net_layers_[i-1].size());
        net_layers_[i].populate(this->r, sqrt(2.00/layer_depth_));
        net_layers_[i].forward(net_layers_[i-1].results_, true);
    }

    net_layers_[layer_count_].init(result_depth_, net_layers_[layer_count_-1].size());
    net_layers_[layer_count_].populate(this->r, sqrt(2.00/layer_depth_));
    net_layers_[layer_count_].forward(net_layers_[layer_count_-1].results_, false);
}






