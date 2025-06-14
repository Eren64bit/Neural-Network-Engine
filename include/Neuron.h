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
    double deltaE(double output, double target);
    void calculateWeight(const std::vector<double>& inpt, double learningRate);
public:
    Neuron(int numInputs);

    double forward(const std::vector<double>& inputV); 
    void train(const std::vector<double>& input, double target, double learningRate); 



    void computeHiddenLayerDelta(const std::vector<std::unique_ptr<Neuron>>& nextLayer, int neuronIndexInPrevLayer);
    void applyWeightUpdate(const std::vector<double>& inpt, double learningRate);



    double readDelta() const {return delta;}
    void setDelta(double i) {this->delta = i;}

    double getWeight(int index) const {return weight[index];}

    
};