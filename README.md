# Neural Network Engine

A simple feedforward neural network engine implemented in C++. This project demonstrates the basics of neural network construction, forward propagation, and backpropagation with support for multiple layers and neurons.

## Features

- Multi-layer neural network architecture
- Customizable number of layers and neurons
- Sigmoid activation function
- Backpropagation learning algorithm
- Memory safety with `std::unique_ptr`
- Example usage for the XOR problem

## Directory Structure

```
Neural Network Engine/
│
├── include/
│   ├── Layer.h
│   ├── Neuron.h
│   └── NeuralNetwork.h
│
├── src/
│   ├── Layer.cpp
│   ├── Neuron.cpp
│   ├── NeuralNetwork.cpp
│   └── main.cpp
```

## Build Instructions

Compile all source files using g++ (C++17 or later):

```sh
g++ -std=c++17 -Iinclude src/*.cpp -o neuralnet
```

## Example Usage

The following example demonstrates how to train the network for the XOR problem:

```cpp
#include "../include/NeuralNetwork.h"

int main() {
    NeuralNetwork net;

    net.addLayer(2, 2); // Input layer: 2 neurons, 2 inputs
    net.addLayer(1);    // Output layer: 1 neuron

    std::vector<std::vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };

    double lr = 0.1;

    for (int epoch = 0; epoch < 10000; epoch++) {
        for (int i = 0; i < inputs.size(); i++) {
            net.kickStartNet(inputs[i], targets[i], lr);
        }
    }

    // Test results
    for (int i = 0; i < inputs.size(); i++) {
        auto out = net.forward(inputs[i]);
        std::cout << "Input: " << inputs[i][0] << ", " << inputs[i][1]
                  << " => Output: " << out[0] << std::endl;
    }

    return 0;
}
```

## License

This project is for educational purposes. Contributions are welcome via pull requests.

---
