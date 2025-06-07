#pragma once
#include "Neuron.h"

class Layer {
private:
    std::vector<std::unique_ptr<Neuron>> neurons;
public:
    Layer(int numNeurons, int numInputsPerNeuron);
    std::vector<double> forward(const std::vector<double>& input);

    void train(const std::vector<double>& input, const std::vector<double>& target, double learningRate);

    int neuronCount() const {return neurons.size();}
    
};