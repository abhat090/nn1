// nn1::layercpp by Arya Bhattacharjee

#include <layer.h>

NeuronLayer::NeuronLayer() {
    this->size_ = -1;
    this->depth_ = -1;
}

void NeuronLayer::init(int s, int d) {
    this->size_ = s;
    this->depth_ = d;

    this->weights_.setZero(s, d);
    this->biases_.setZero(s);
    this->z_.setZero(s);
    this->results_.setZero(s);
}


void NeuronLayer::populate(RanD &r, double scale) {
    for(double& d : this->weights_.reshaped()) {
        d = r.gen() * scale;
    }
}

void NeuronLayer::forward(const Eigen::VectorXd &v, bool relu) {
    this->z_ = (this->weights_*v) + this->biases_;
    for(int j = 0; j < this->size_; j++) {
        this->results_[j] =  relu ? ReLU(this->z_[j]) : Sigmoid(this->z_[j]);
    }
}
