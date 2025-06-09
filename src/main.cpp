#include "../include/Neuron.h"
#include "../include/NeuralNetwork.h"
#include "../include/Layer.h"

int main() {
    NeuralNetwork net;

    net.addLayer(2, 2); // input layer: 2 input, 2 nöron
    net.addLayer(1);    // output layer: 1 nöron (giriş otomatik algılanıyor)

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

    // Test sonuçları
    for (int i = 0; i < inputs.size(); i++) {
        auto out = net.forward(inputs[i]);
        std::cout << "Input: " << inputs[i][0] << ", " << inputs[i][1] 
                  << " => Output: " << out[0] << std::endl;
    }

    return 0;
}
