#include "../include/Layer.h"

Layer::Layer(int numNeurons, int numInputsPerNeuron) {

    for (int i = 0; i < numNeurons; i++) {
        
        neurons.push_back(std::make_unique<Neuron>(numInputsPerNeuron));
    }
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    lastOutput.clear();
    for (const auto& neuron : neurons) {
        lastOutput.push_back(neuron->forward(input));
    }
    return lastOutput;
}

std::vector<double> Layer::computeOutputLayerDeltas(const std::vector<double>& target) {
    std::vector<double> deltas;
    for (int i = 0; i < neurons.size(); i++) {
        double output = lastOutput[i];
        double error = output - target[i];
        double delta = error * output * (1 - output);
        neurons[i]->setDelta(delta);
        deltas.push_back(delta);
    }
    return deltas;
}


void Layer::computeHiddenLayerDeltas(const Layer& nexLayer) {
    for (int i = 0; i < neurons.size(); i++) {
        neurons[i]->computeHiddenLayerDelta(nexLayer.neurons, i);
    }
}

void Layer::applyWeightUpdate(const std::vector<double>& input, double learningRate) {
    for (int i = 0; i < neurons.size(); i++) {
        neurons[i]->applyWeightUpdate(input, learningRate);
    }
}

