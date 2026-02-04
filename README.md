# nn1 (Primitive Deep Neural Network)

>Authored by: [Arya Bhattacharjee](https://github.com/abhat090)

This is a very simple/primitive neural network designed to run relatively quickly when run on a multithreaded CPU.

There is plenty of room for improvement, as of the current version, the network achieved a maximum of 96% accuracy when trained and tested on the MNIST dataset (3 layers of 32 neurons).

It implements a fairly standard backpropagation algorithm, using ReLU (Rectifier) for the hidden layer activation functions, and Sigmoid for the output layer activation function. The algorithm also implements an L2 regularization for backpropagation in order to prevent the ReLU function from "blowing up" the entire network.

The task of training is divided into multiple threads, each running one training image through a "cycle", consisting of forwarding the image, and then calculating the cost-gradient from that image using backpropagation. These cycles are run in mini-batches of 128 (default) for the entirety of the training dataset. Additionally, the model is run for multiple epochs, where the training data is shuffled, and the mini-batches are recreated.

Feel free to download and mess around with the code, I'm sure there's plenty of room for improvement.
## Build
The project has been configured and build using CMake;
```CMake
cmake --build build
```
