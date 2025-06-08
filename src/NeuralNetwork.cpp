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

std::vector<double> NeuralNetwork::kickStartNet(const std::vector<double>& input, const std::vector<double>& target,double learningRate) {
    std::vector<std::vector<double>> activision;
    activision.push_back(input);
    for (int i = 0; i < NeuralNet.size(); i++) {
        activision.push_back(NeuralNet[i]->forward(activision[i]));
        if (i == NeuralNet.size() - 1) NeuralNet[i]->train(activision[i], target, learningRate);
    }
    return activision.back();
}

