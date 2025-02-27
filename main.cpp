// nn1 by Arya Bhattacharjee

/*
 * A very primitive deep neural network.
 * Current best accuracy (mnist) : 95%
 */

#include <iostream>
#include <tuple>
#include <chrono>
#include <future>

#include "layer.h"
#include "network.h"
#include "mnist/mnist_reader.h"

/*
 * Multithreaded process (CPU)
 * Ensure that threads are a power of 2 in order to divide batches properly
 */
#define THREADS 8

#define TRAINING_SIZE 59904
#define BATCH_SIZE 128
#define EPOCHS 20
#define TESTING_SIZE 10000

#define LAYER_COUNT 3
#define LAYER_DEPTH 32
#define LEARN_RATE 0.005
#define RESULT_DEPTH 10

using namespace std;

Network network;

MnistReader mnist_training(TRAINING_SIZE, "/train-images.idx3-ubyte", "/train-labels.idx1-ubyte");
MnistReader mnist_testing(TESTING_SIZE, "/t10k-images.idx3-ubyte", "/t10k-labels.idx1-ubyte");

// Parallelized batch operations
/*
 * Each batch runs its backpropagation cycles in seperate threads.
 * The results from each thread are then summed into the network.
 */
void net_batch(int begin, int end, int batchId) {
    vector<Eigen::MatrixXd> del_w; // weights
    vector<Eigen::VectorXd> del_b; // biases

    vector<std::future<gradient>> cycle_threads(THREADS);
    vector<gradient> cycle_results(THREADS);

    for (NeuronLayer& nl : network.layers()) {
        del_w.push_back(Eigen::MatrixXd().setZero(nl.weights_.rows(), nl.weights_.cols()));
        del_b.push_back(Eigen::VectorXd().setZero(nl.biases_.size()));
    }

    auto start = chrono::steady_clock::now();

    for (int i = begin; i < end; i+=THREADS) {
        // Initialize each thread with the Network cycle function
        for (std::future<gradient>& thrd : cycle_threads) {
            thrd = std::async(&Network::cycle, &network,
                mnist_training.getImage(i).reshaped(),
                yVec((int) mnist_training.getLabel(i), RESULT_DEPTH, 1));
        }

        // Obtain results from parallel processes
        int thread_ = 0;
        for (gradient& c_g: cycle_results) {
            c_g = cycle_threads[thread_].get();
            thread_++;
        }

        // Sum gradients together and store in vector
        u_long j = del_w.size()-1;
        for (Eigen::MatrixXd& m : del_w) {
            for (gradient c_w: cycle_results) m += get<0>(c_w)[j];
            j--;
        }

        u_long k = del_b.size()-1;
        for (Eigen::VectorXd& v : del_b) {
            for (gradient c_b: cycle_results) v += get<1>(c_b)[k];
            k--;
        }
    }

    // Apply function to update network weights
    network.apply(del_w, del_b, BATCH_SIZE);

    auto stop = chrono::steady_clock::now();
    auto time = stop-start;
    cout << "Batch " << batchId << ": "
        << chrono::duration<double, std::milli>(time).count() << " ms | Res: "
        << network << endl;
}

/*
 * Simple test function for network:
 * Forwards test image, checks correctness and returns # correct
 */
int test() {
    int correct = 0;

    for (int i = 0; i < TESTING_SIZE; i++) {
        const int label = (int) mnist_testing.getLabel(i);
        const Eigen::MatrixXd image = mnist_testing.getImage(i).reshaped();

        if (network.forward(image) == label) correct++;
    }

    return correct;
}

int main() {
    // Configure Network
    network.setLayerCount(LAYER_COUNT);
    network.setLayerDepth(LAYER_DEPTH);
    network.setLearnRate(LEARN_RATE);
    network.setResultDepth(RESULT_DEPTH);

    // Get any image to initialize network with random values
    Eigen::VectorXd image = mnist_training.getImage(0).reshaped();
    network.init(image, mnist_training.image_size());

    /*
     * Each epoch shuffles the training data and
     * runs batches through the entire dataset
     *
     * Training size controls home much of the actual dataset
     * should be used for training (total cycles per epoch)
     */
    int epoch = 0;
    while(epoch < EPOCHS) {
        mnist_training.shuffle(epoch + 12);

        int batchId = 0;
        for (int i = 0; i < TRAINING_SIZE - BATCH_SIZE; i+=BATCH_SIZE) {
            net_batch(i, i+BATCH_SIZE-1, ++batchId);
        }

        // Run a test after every epoch to see improvements
        int correct = test();
        cout << endl << "Epoch " << epoch << " complete: "
        << correct << "/" << TESTING_SIZE << " | "
        << (correct * 100)/TESTING_SIZE << "%" << endl;
        epoch++;
    }
}
