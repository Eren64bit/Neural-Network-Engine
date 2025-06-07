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
    double bias = 0.0;

    double dot(const std::vector<double>& a, const std::vector<double>& b);
    double sigmoid(double a);
    double delta(double output, double target); // düzeltildi
public:
    Neuron(int numInputs);

    double forward(const std::vector<double>& inputV); // düzeltildi
    void train(const std::vector<double>& input, double target, double learningRate); // düzeltildi
};