/** Copyright (C) 2023 briand (https://github.com/briand-hub)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "BriandNN.hxx"

using namespace std;

Briand::Neuron::Neuron() {
    this->Inputs = make_unique<vector<unique_ptr<Synapsis>>>();
}

Briand::Neuron::Neuron(const double& value) : Neuron() {
    this->Value = value;
    // ID to zero
    this->Id = 0;
}

void Briand::Neuron::UpdateValue(ActivationFunction activationFunction) {
    // Nothing to do?
    if (this->Inputs == nullptr || this->Inputs->size() == 0) return;

    // Reset the current value
    this->Value = 0.0;

    // For each input synapsis
    for (auto syn = this->Inputs->begin(); syn != this->Inputs->end(); syn++) {
        // Update value (weighted sum)
        this->Value += syn->get()->Source->Value * syn->get()->Weight;
    }

    // After that, activate
    this->Value = activationFunction(this->Value);
}

void Briand::Neuron::ConnectTo(const unique_ptr<Neuron>& other, double weight /*= 1.0*/) {
    if (other == nullptr) throw runtime_error("Briand::Neuron::ConnectTo - cannot connect to nothing.");

    // Create a new Synapsis (I am the source)
    auto syn = make_unique<Synapsis>();
    syn->Source = this; 
    syn->Weight = weight;

    // Add synapsis (connect this neuron to the other)
    other->Inputs->push_back(std::move(syn));

    //
    // TODO: Should be verified that same connection is not existing!
    //
}

Briand::NeuralLayer::NeuralLayer(const LayerType& type, ActivationFunction activationFunction) {
    // Initialize neurons with empty vector
    this->Neurons = make_unique<vector<unique_ptr<Neuron>>>();
    
    // Save the activation function
    this->_activationFunction = activationFunction;
}

void Briand::NeuralLayer::UpdateNeurons() {
    // For each neuron inside this layer, update value and activate
    for (auto neuron = this->Neurons->begin(); neuron != this->Neurons->end(); neuron++) {
        neuron->get()->UpdateValue(this->_activationFunction);
    }
}

Briand::NeuralNetwork::NeuralNetwork() {
    // Do not do anything!
}

void Briand::NeuralNetwork::PropagateForward() {
    // This must calculate values from inputs to outputs.

    // If no input or output layer has neurons, throw an error
    if (this->InputLayer == nullptr || this->InputLayer->Neurons->size() == 0) throw runtime_error("Briand::NeuralNetwork::PropagateForward - no inputs");
    if (this->OutputLayer == nullptr || this->OutputLayer->Neurons->size() == 0) throw runtime_error("Briand::NeuralNetwork::PropagateForward - no outputs");

    // The UpdateValue method is meant to be "update my value with my inputs" for each neuron. So we have to start from the first
    // hidden layer or, if nothing, from the output layer

    if (this->HiddenLayers != nullptr && this->HiddenLayers->size() > 0) {
        for (auto layer = this->HiddenLayers->begin(); layer != this->HiddenLayers->end(); layer++) {
            // In layer, calculate all neurons value
            layer->get()->UpdateNeurons();
        }
    } 

    // Latest layer is output
    this->OutputLayer->UpdateNeurons();

    // After that output layer neurons will have the output value
}

Briand::Perceptron::Perceptron(const int& inputs, ActivationFunction activationFunction) {
    // Inputs must be valid
    if (inputs <= 0) throw runtime_error("Briand::Perceptron::Perceptron - inputs must be greater than 0.");

    // No hidden layers
    this->HiddenLayers.reset();

    // Prepare output with 1 neuron
    this->OutputLayer = make_unique<NeuralLayer>(LayerType::Output, activationFunction);
    auto out = make_unique<Neuron>();
    out->Value = 0.0;
    
    this->InputLayer = make_unique<NeuralLayer>(LayerType::Input, activationFunction);
    for (int i = 0; i<inputs; i++) {
        // Create a new neuron connected with output

        auto in = make_unique<Neuron>(1.0);
        in->ConnectTo(out, 1.0);

        this->InputLayer->Neurons->push_back(std::move(in));
    }

    this->OutputLayer->Neurons->push_back(std::move(out));
}

void Briand::Perceptron::PropagateForward() {
    // Call the base
    NeuralNetwork::PropagateForward();
}

double Briand::Perceptron::GetResult() {
    this->PropagateForward();

    // Return neuron value in the output layer
    return this->OutputLayer->Neurons->begin()->get()->Value;
}

void Briand::Perceptron::PropagateBackward(const double& target) {
    //
    // TODO
    //
    throw runtime_error("UNIMPLEMENTED");
}



