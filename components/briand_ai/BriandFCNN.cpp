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

#include "BriandFCNN.hxx"

using namespace std;
using namespace Briand;

/**********************************************************************
    Neural Layer class
***********************************************************************/

NeuralLayer::NeuralLayer(const LayerType& type, const size_t& neurons, ActivationFunction f, ActivationFunction df, ErrorFunction e) {
    // Check
    if (neurons == 0) throw out_of_range("Neurons must be > 0 for any layer");
    if (type == LayerType::Input && (f != nullptr || df != nullptr || e != nullptr)) throw runtime_error("Cannot specify f, df or e for input layer!");
    if (type != LayerType::Input && (f == nullptr || df == nullptr)) throw runtime_error("Must specify f and df for non-input layers!");
    if (type == LayerType::Output && e == nullptr) throw runtime_error("Must specify cost/error calculation for output layer!");
    if (type != LayerType::Output && e != nullptr) throw runtime_error("Cannot specify cost/error calculation for non-output layers!");

    // Initialize
    this->_f = f;
    this->_df = df;
    this->_E = e;
    this->_type = type;
    this->_weights = nullptr;

    // Bias neuron value is always 1 so just handle the weights (FCN)
    this->_bias_weights = nullptr;
    if (this->_type == LayerType::Input || this->_type == LayerType::Hidden) {
        // Initialize all weights to 1
        this->_bias_weights = make_unique<vector<double>>(neurons, 1.0);
    }

    this->_neuronsNet = make_unique<vector<double>>();
    this->_neuronsNet->reserve(neurons);

    this->_neuronsOut = make_unique<vector<double>>();
    this->_neuronsOut->reserve(neurons);

    // Initialize neurons to 0
    for (size_t i=0; i<neurons; i++) {
        this->_neuronsNet->push_back(0.0);
        this->_neuronsOut->push_back(0.0);
    }
}

NeuralLayer::NeuralLayer(const LayerType& type, const size_t& neurons, ActivationFunction f, ActivationFunction df, ErrorFunction e, const Matrix& weights) 
    : NeuralLayer(type, neurons, f, df, e)
{
    // Weights allowed for non-input layers 
    if (this->_type == LayerType::Input) throw runtime_error("Weights not allowed for input layer.");

    // Weight matrix rows must be equal to layer's neurons (not including bias)
    if (weights.Rows() != neurons) throw runtime_error("Weight matrix invalid: must have as many rows as layer's neurons!");

    // Weight matrix cols must be equal to layer's input (cannot check there)

    this->_weights = make_unique<Matrix>(weights);
}

NeuralLayer::NeuralLayer(const LayerType& type, const size_t& neurons, ActivationFunction f, ActivationFunction df, ErrorFunction e, const std::initializer_list<std::initializer_list<double>>& weights)
    : NeuralLayer(type, neurons, f, df, e, Matrix{weights})
{
}

NeuralLayer::~NeuralLayer() {
    this->_weights.reset();
    this->_neuronsNet.reset();
    this->_neuronsOut.reset();
}

void NeuralLayer::SetBiasWeights(const vector<double>& bias_weights) { 
    // Allowed only for input or hidden layer
    if (this->_type != LayerType::Hidden && this->_type != LayerType::Input) throw runtime_error("Bias allowed only for input or hidden layer.");

    this->_bias_weights = make_unique<vector<double>>(bias_weights); 
}

/**********************************************************************
    FCNN class
***********************************************************************/

FCNN::FCNN() {
    this->_hasOutputs = false;
    this->_layers = make_unique<vector<unique_ptr<NeuralLayer>>>();
}

FCNN::~FCNN() {
    this->_layers.reset();
}

void FCNN::AddInputLayer(const size_t& inputs) {
    // Check
    if (this->_layers->size() > 0) throw runtime_error("Input layer has been added before.");

    auto layer = make_unique<NeuralLayer>(LayerType::Input, inputs, nullptr, nullptr, nullptr);
    this->_layers->push_back(std::move(layer));
}

void FCNN::AddInputLayer(const size_t& inputs, const vector<double>& values) {
    // Check
    if (values.size() != inputs) throw runtime_error("Input values: invalid size.");

    this->AddInputLayer(inputs);

    // Initialize values
    for (int i = 0; i<inputs; i++) this->_layers->at(0)->_neuronsOut->at(i) = values[i];
}

void FCNN::SetInput(const vector<double>& values) {
    // Check
    if (this->_layers == nullptr || this->_layers->size() < 1) throw runtime_error("Cannot set input values: missing input layer.");
    if (values.size() != this->_layers->at(0)->_neuronsOut->size()) throw runtime_error("Input values: invalid size.");

    for (int i = 0; i<this->_layers->at(0)->_neuronsOut->size(); i++) this->_layers->at(0)->_neuronsOut->at(i) = values[i];
}

void FCNN::AddHiddenLayer(const size_t& neurons, const ActivationFunction& activationFunc, const ActivationFunction& activationDer) {
    // Check
    if (this->_layers == nullptr || this->_layers->size() < 1) throw runtime_error("Cannot add hidden layer: missing an input layer.");
    if (this->_hasOutputs) throw runtime_error("Cannot add hidden layer after output layer!");

    // Default weights matrix with 1.0 value, as many rows as neurons, as many columns as previous layer neurons.
    const int rows = neurons;
    const int cols = this->_layers->at(this->_layers->size() - 1)->_neuronsOut->size();

    auto layer = make_unique<NeuralLayer>(LayerType::Hidden, neurons, activationFunc, activationDer, nullptr, Matrix{rows, cols, 1.0});
    this->_layers->push_back(std::move(layer));
}

void FCNN::AddHiddenLayer(const size_t& neurons, const ActivationFunction& activationFunc, const ActivationFunction& activationDer, const Matrix& weights) {
    // Check
    if (this->_layers == nullptr || this->_layers->size() < 1) throw runtime_error("Cannot add hidden layer: missing an input layer.");
    if (this->_hasOutputs) throw runtime_error("Cannot add hidden layer after output layer!");

    // Check: matrix must have as many rows as the current layer neurons
    if (neurons != weights.Rows()) throw out_of_range("Invalid weights: weight matrix rows must be equal to the number of this layer neurons.");

    // Check: matrix must have as many columns as the PREVIOUS layer neurons
    if (this->_layers->at(this->_layers->size() - 1)->_neuronsOut->size() != weights.Cols()) throw out_of_range("Invalid weights: weight matrix cols must be equal to the number of previous layer neurons.");

    auto layer = make_unique<NeuralLayer>(LayerType::Hidden, neurons, activationFunc, activationDer, nullptr, weights);
    this->_layers->push_back(std::move(layer));
}

void FCNN::AddOutputLayer(const size_t& outputs, const ActivationFunction& activationFunc, const ActivationFunction& activationDer, const ErrorFunction& errorFunc) {
    // Check
    if (this->_hasOutputs) throw runtime_error("Output layer has been added before.");
    if (this->_layers == nullptr || this->_layers->size() < 1) throw runtime_error("Cannot add output layer: missing an input layer.");

    // Default weights matrix with 1.0 value, as many rows as neurons, as many columns as previous layer neurons.
    const int rows = outputs;
    const int cols = this->_layers->at(this->_layers->size() - 1)->_neuronsOut->size();

    auto layer = make_unique<NeuralLayer>(LayerType::Output, outputs, activationFunc, activationDer, errorFunc, Matrix{rows, cols, 1.0});
    this->_layers->push_back(std::move(layer));

    // Close network build
    this->_hasOutputs = true;
}

void FCNN::AddOutputLayer(const size_t& outputs, const ActivationFunction& activationFunc, const ActivationFunction& activationDer, const ErrorFunction& errorFunc, const Matrix& weights) {
    // Check
    if (this->_hasOutputs) throw runtime_error("Output layer has been added before.");
    if (this->_layers == nullptr || this->_layers->size() < 1) throw runtime_error("Cannot add output layer: missing an input layer.");

    // Check: matrix must have as many rows as the current layer neurons
    if (outputs != weights.Rows()) throw out_of_range("Invalid weights: weight matrix rows must be equal to the number of this layer neurons.");

    // Check: matrix must have as many columns as the PREVIOUS layer neurons
    if (this->_layers->at(this->_layers->size() - 1)->_neuronsOut->size() != weights.Cols()) throw out_of_range("Invalid weights: weight matrix cols must be equal to the number of previous layer neurons.");


    auto layer = make_unique<NeuralLayer>(LayerType::Output, outputs, activationFunc, activationDer, errorFunc, weights);
    this->_layers->push_back(std::move(layer));

    // Close network build
    this->_hasOutputs = true;
}

void FCNN::Propagate() {
    // Check
    if (this->_layers == nullptr || this->_layers->size() < 1) throw runtime_error("Cannot propagate: missing an input layer.");
    if (!this->_hasOutputs) throw runtime_error("Cannot propagate: missing an output layer.");
    if (this->_layers == nullptr || this->_layers->size() < 2) throw runtime_error("Cannot propagate with less than 2 layers!");

    // Weighted sum calculation, starting from the first layer after input.
    for (auto it = this->_layers->begin() + 1; it != this->_layers->end(); it++) {
        // Previous layer l-1
        const auto& l_1 = (it - 1)->get();
        // Current layer a_(l)
        const auto& l = it->get();

        // If the previous layer is the input layer, separate calculus to add bias
        // (backpropagating would drive to wrong input value if iterated)

        if (l_1->_type == LayerType::Input && l_1->_bias_weights != nullptr && l_1->_bias_weights->size() > 0) {
            // copy values
            vector<double> a_l_1;
            a_l_1.assign(l_1->_neuronsOut->begin(), l_1->_neuronsOut->end()); 
            // add biasing
            for (size_t i = 0; i<a_l_1.size(); i++) a_l_1[i] += l_1->_bias_weights->at(i);

            // Weighted sum can be performed with weight_matrix * vector
            // In math: z_(l) = W_(l) * a_(l-1)
            l->_neuronsNet = l->_weights->MultiplyVector(a_l_1);
        }
        else {
            // Direct, save memory
            
            // Weighted sum can be performed with weight_matrix * vector
            // In math: z_(l) = W_(l) * a_(l-1)
            l->_neuronsNet = l->_weights->MultiplyVector(*l_1->_neuronsOut.get());
        }    

        // Now activate neurons applying the activation function of this layer
        // In math a_l = f(z_l)
        for (int i = 0; i< l->_neuronsNet->size(); i++) {
            // If current layer has a bias, add the weighted value (1*b_i) to each neuron
            if (l->_bias_weights != nullptr) l->_neuronsNet->at(i) += l->_bias_weights->at(i);
            
            // Activate
            l->_neuronsOut->at(i) = l->_f( l->_neuronsNet->at(i) );
        }
    }
}

unique_ptr<vector<double>> FCNN::GetResult() {
    // Check
    if (!this->_hasOutputs) throw runtime_error("GetResult() Error: missing an output layer.");

    auto& out = this->_layers->at(this->_layers->size() - 1);
    auto result = make_unique<vector<double>>();
    result->assign(out->_neuronsOut->begin(), out->_neuronsOut->end());

    return std::move(result);
}

unique_ptr<vector<double>> FCNN::Predict(const vector<double>& inputs) {
      // Set inputs and propagate forward
    this->SetInput(inputs);
    this->Propagate();

    // Get results
    return this->GetResult();
}

double FCNN::Train(const vector<double>& inputs, const vector<double>& targets) {
    // Check
    if (this->_layers == nullptr || this->_layers->size() < 1) throw runtime_error("Cannot backpropagate: missing an input layer.");
    if (!this->_hasOutputs) throw runtime_error("Cannot backpropagate: missing an output layer.");
    if (targets.size() != this->_layers->at(this->_layers->size() - 1)->_neuronsOut->size()) throw out_of_range("Invalid targets: size must be equal to outputs.");

    // Get results
    auto outputs = this->Predict(inputs);
    auto& outputLayer = this->_layers->at(this->_layers->size() - 1);

    double totalError = 0;

    // Calculate errors at output and total error
    auto E = make_unique<vector<double>>(targets);
    for(size_t i = 0; i<E->size(); i++) {
        E->at(i) = outputLayer->_E(targets[i], outputs->at(i)); 
        totalError += E->at(i);
    }


    

    return totalError;
}

void FCNN::PrintResult() {
    // Check
    if (!this->_hasOutputs) throw runtime_error("GetResult() Error: missing an output layer.");

    auto& out = this->_layers->at(this->_layers->size() - 1);
    printf("| ");
    for (auto it = out->_neuronsOut->begin(); it != out->_neuronsOut->end(); it++)
        printf(" %lf ", *it);
    printf(" |\n");
}