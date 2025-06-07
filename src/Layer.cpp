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

void Layer::train(const std::vector<double>& input, const std::vector<double>& target, double learningRate) {
    if (neurons.size() != target.size()) throw std::runtime_error("error: input and target size does not match up");
    for (int i = 0; i < input.size(); i++) {
        neurons[i]->train(input, target[i], learningRate);
    }
}