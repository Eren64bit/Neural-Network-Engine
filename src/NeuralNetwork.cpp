#include "../include/NeuralNetwork.h"


void NeuralNetwork::addLayer(int numNeurons, int inputCount) {
    if (NeuralNet.empty()) {
        if (inputCount == -1) throw std::runtime_error("First layer input count must be specified!");
        inputSize = inputCount;
        NeuralNet.push_back(std::make_unique<Layer>(numNeurons, inputSize));
    } else {
        const Layer& lastLayer = *NeuralNet.back();
        NeuralNet.push_back(std::make_unique<Layer>(numNeurons, lastLayer.neuronCount()));
    }
}


std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> output;
    output = NeuralNet[0]->forward(input);
    for (int i = 1; i < NeuralNet.size(); i++) {
        output = NeuralNet[i]->forward(output);
    }
    return output;
}

/*İlk olarak giriş input vektörünü alır.

İlk Layer'a gönderir → layer forward işlemi yapar → bir output üretir.

Bu output, sıradaki layer’a input olarak gönderilir.

Bu işlem tüm layer’lar bitene kadar devam eder.

Sonucun çıktısı return edilir.*/