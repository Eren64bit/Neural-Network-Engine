#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <random>

class Neuron {
private:
    std::vector<double> input;
    std::vector<double> weight;
    double output_i;
    double sigmoid_i;
    double delta;
    double bias = 0.0;

    double dot(const std::vector<double>& a, const std::vector<double>& b);
    double sigmoid(double a);
    double deltaE(double output, double target); // düzeltildi
public:
    Neuron(int numInputs);

    double forward(const std::vector<double>& inputV); // düzeltildi
    void train(const std::vector<double>& input, double target, double learningRate); // düzeltildi
    void computeHiddenLayerDelta(const std::vector<std::unique_ptr<Neuron>>& nextLayer, int neuronIndexInPrevLayer);
    double readDelta() const {return delta;}
    double getWeight(int index) const {return weight[index];}
};