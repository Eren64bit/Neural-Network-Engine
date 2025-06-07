#include "../include/Layer.h"

Layer::Layer(int numNeurons, int numInputsPerNeuron) {

    for (int i = 0; i < numNeurons; i++) {
        
        neurons.push_back(std::make_unique<Neuron>(numInputsPerNeuron));
    }
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    std::vector<double> outputs;
    for (const auto& neuron : neurons) {
        outputs.push_back(neuron->forward(input));
    }
    return outputs;
}