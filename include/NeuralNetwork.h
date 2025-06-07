#pragma once
#include "Neuron.h"
#include "Layer.h"

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> NeuralNet;  
    int inputSize = 0;
    
public:
    void addLayer(int numNeuron, int inputCount = -1);

    std::vector<double> forward(const std::vector<double>& input);
};