#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <random>

//δ=(output−target)⋅output⋅(1−output)
class Neuron {
private:
    std::vector<double> input;
    std::vector<double> weight;
    double bias = 0.0;

    double dot(std::vector<double>& a, std::vector<double>& b);
    double sigmoid(double a);
    double delta(double target, double output);
public:
    Neuron(int numInputs);

    double forward(std::vector<double> inputV); //dP stand for dot producted
    void train(const std::vector<double>& input, double target, double learningRate);
};