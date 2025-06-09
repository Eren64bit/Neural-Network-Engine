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

std::vector<double> NeuralNetwork::kickStartNet(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
    std::vector<std::vector<double>> activision;
    activision.push_back(input);

    // Forward pass
    for (int i = 0; i < NeuralNet.size(); i++) {
        activision.push_back(NeuralNet[i]->forward(activision[i]));
    }

    // Output layer delta + weight update
    int lastIdx = NeuralNet.size() - 1;
    NeuralNet[lastIdx]->computeOutputLayerDeltas(target);
    NeuralNet[lastIdx]->applyWeightUpdate(activision[lastIdx], learningRate);

    // Hidden layers (geri doğru)
    for (int i = lastIdx - 1; i >= 0; i--) {
        NeuralNet[i]->computeHiddenLayerDeltas(*NeuralNet[i + 1]);
        NeuralNet[i]->applyWeightUpdate(activision[i], learningRate);
    }

    return activision.back(); // output
}


