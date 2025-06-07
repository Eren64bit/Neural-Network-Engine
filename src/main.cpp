#include "../include/Neuron.h"
#include "../include/NeuralNetwork.h"
#include "../include/Layer.h"


int main() {
    NeuralNetwork test;
    test.addLayer(3, 2);
    test.addLayer(1);
    std::vector<double> result = test.forward({0.5, 0.8});

    for (int i = 0; i < result.size(); i++) {
        std::cout << result[i] << '\n';
    }

    return 0;
}
