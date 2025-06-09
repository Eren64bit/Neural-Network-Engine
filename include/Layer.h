#pragma once
#include "Neuron.h"

class Layer {
private:
    std::vector<std::unique_ptr<Neuron>> neurons;
    std::vector<double> lastOutput;
public:
    Layer(int numNeurons, int numInputsPerNeuron);
    std::vector<double> forward(const std::vector<double>& input);


    std::vector<double> computeOutputLayerDeltas(const std::vector<double>& target);
    void computeHiddenLayerDeltas(const Layer& nexLayer);

    void applyWeightUpdate(const std::vector<double>& input, double learningRate);

    int neuronCount() const {return neurons.size();}
    
};