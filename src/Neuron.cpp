#include "../include/Neuron.h"

//δ=(output−target)⋅output⋅(1−output)

Neuron::Neuron(int inpt) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for (int i = 0; i < inpt; i++) {
        weight.push_back(dist(gen));
    }

    bias = dist(gen);
}

double Neuron::sigmoid(double x) {
    return (1/(1 + (exp(-x))));
}

double Neuron::delta(double output, double target) {
    return (output - target) * output * (1 - output);
}

double Neuron::dot(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }
    return result;
}

double Neuron::forward(const std::vector<double>& inputV) {
    if (inputV.size() != weight.size())
        throw std::runtime_error("Input and weight sizes do not match!");

    double sum = dot(inputV, weight) + bias;
    return sigmoid(sum);
}

void Neuron::train(const std::vector<double>& inpt, double target, double learningRate) {
    if (inpt.size() != weight.size())
        throw std::runtime_error("error: Cannot train neuron input and weight size does not match up");

    double output = forward(inpt);
    double dlt = delta(output, target);

    for (int i = 0; i < weight.size(); i++) {
        weight[i] -= learningRate * dlt * inpt[i];
    }

    bias -= learningRate * dlt;
}
