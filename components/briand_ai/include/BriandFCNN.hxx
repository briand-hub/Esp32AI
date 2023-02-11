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

#pragma once

#ifndef BRIAND_FCNN_H
#define BRIAND_FCNN_H

#include "BriandInclude.hxx"
#include "BriandMatrix.hxx"
#include "BriandMath.hxx"

using namespace std;
using namespace Briand;

namespace Briand {

    /** @brief A layer of neurons */
    class NeuralLayer {
        protected:

        /// @brief Weights FROM PREVIOUS LAYER
        unique_ptr<Matrix> _weights;

        /// @brief Neuron net values 
        unique_ptr<vector<double>> _neuronsNet;

        /// @brief Neuron activated values 
        unique_ptr<vector<double>> _neuronsOut;
        
        /// @brief Bias neuron weight (input and hidden layers only, otherwise 0)
        double _bias_weight;

        /// @brief Layer type
        LayerType _type;

        /// @brief Layer activation function (hidden and output layer only)
        ActivationFunction _f;

        /// @brief Layer activation function derivative (hidden and output layer only)
        ActivationFunction _df;

        /// @brief Error calculation function
        ErrorFunction _E;

        public:

        /// @brief Builds a layer.
        /// @param type Layer type
        /// @param neurons Number of neurons
        /// @param f Activation function (hidden and output layer only, mandatory)
        /// @param df Activation function derivative (hidden and output layer only, mandatory)
        /// @param e Error/Cost function (output layer only, required)
        NeuralLayer(const LayerType& type, const unsigned int& neurons, ActivationFunction f, ActivationFunction df, ErrorFunction e);

        /// @brief Builds a layer with specified weights.
        /// @param type Layer type
        /// @param neurons Number of neurons
        /// @param f Activation function (hidden and output layer only, mandatory)
        /// @param df Activation function derivative (hidden and output layer only, mandatory)
        /// @param e Error/Cost function (output layer only, required)
        /// @param weights Weights to the next layer (input and hidden layers only). 1 row for each layer's neuron, 1 column for each previous layer neuron.
        NeuralLayer(const LayerType& type, const unsigned int& neurons, ActivationFunction f, ActivationFunction df, ErrorFunction e, const Matrix& weights);

        /// @brief Builds a layer with specified weights.
        /// @param type Layer type
        /// @param neurons Number of neurons
        /// @param f Activation function (hidden and output layer only, mandatory)
        /// @param df Activation function derivative (hidden and output layer only, mandatory)
        /// @param e Error/Cost function (output layer only, required)
        /// @param weights Weights to the next layer (input and hidden layers only). 1 row for each layer's neuron, 1 column for each previous layer neuron.
        NeuralLayer(const LayerType& type, const unsigned int& neurons, ActivationFunction f, ActivationFunction df, ErrorFunction e, const std::initializer_list<std::initializer_list<double>>& weights);

        ~NeuralLayer();

        /// @brief Set the error calculation function (output layer only)
        /// @param fError Error calculation function
        void SetOutputErrorAs(const ErrorFunction& fError);

        /// @brief Set the bias weight (input and hidden layers only)
        /// @param bias_weight The bias weight (value always 1)
        void SetBiasWeight(const double& bias_weight);

        /* The FCNN class can access to all properties and methods */
        friend class FCNN;
    }; 

    /// @brief An empty Neural Network, without layers, neurons and connections.
    /// Has no particular methods, just basic data structure and propagation forward.
    /// Use it when you know what you are doing!
    class FCNN {
        protected:

        unique_ptr<vector<unique_ptr<NeuralLayer>>> _layers;

        bool _hasOutputs;

        public:
        
        /// @brief Build empty FCNN
        FCNN();

        ~FCNN();

        /// @brief Adds input layer (can be called only once). STARTS THE NETWORK CREATION (must be first layer)
        /// @param inputs Number of inputs
        void AddInputLayer(const unsigned int& inputs);

        /// @brief Adds input layer with values (can be called only once). STARTS THE NETWORK CREATION (must be first layer)
        /// @param inputs Number of inputs
        /// @param values Initial input values
        void AddInputLayer(const unsigned int& inputs, const vector<double>& values);

        /// @brief Set input for FCNN
        /// @param values Input values
        void SetInput(const vector<double>& values);
 
        /// @brief Adds hidden layer, in sequence. CONTINUES NETWORK CREATION (must be a "middle" layer)
        /// @param outputs Number of neurons
        /// @param activationFunc Activation function
        /// @param activationDer Activation function derivative
        void AddHiddenLayer(const unsigned int& neurons, const ActivationFunction& activationFunc, const ActivationFunction& activationDer);

        /// @brief Adds hidden layer, in sequence. CONTINUES NETWORK CREATION (must be a "middle" layer)
        /// @param outputs Number of outputs
        /// @param activationFunc Activation function
        /// @param activationDer Activation function derivative
        /// @param weights Weights from previous layer (must have 1 row for each layer's neuron, 1 column for each previous layer neuron)
        void AddHiddenLayer(const unsigned int& neurons, const ActivationFunction& activationFunc, const ActivationFunction& activationDer, const Matrix& weights);

        /// @brief Adds output layer (can be called only once). CLOSES THE NETWORK CREATION (must be latest layer)
        /// @param outputs Number of outputs
        /// @param activationFunc Activation function
        /// @param activationDer Activation function derivative
        /// @param errorFunc Error/cost function
        void AddOutputLayer(const unsigned int& outputs, const ActivationFunction& activationFunc, const ActivationFunction& activationDer, const ErrorFunction& errorFunc);
        
        /// @brief Adds output layer (can be called only once) with weights. CLOSES THE NETWORK CREATION (must be latest layer)
        /// @param outputs Number of outputs
        /// @param activationFunc Activation function
        /// @param activationDer Activation function derivative
        /// @param errorFunc Error/cost function
        /// @param weights Weights from previous layer (must have 1 row for each layer's neuron, 1 column for each previous layer neuron)
        void AddOutputLayer(const unsigned int& outputs, const ActivationFunction& activationFunc, const ActivationFunction& activationDer, const ErrorFunction& errorFunc, const Matrix& weights);
    
        /// @brief Propagates (forward).
        void Propagate();

        /// @brief Back propagation with specified targets.
        /// @param targets The expected output values (must be equal in size to output neurons!)
        void Backpropagate(const vector<double>& targets);

        /// @brief Returns output neurons values after a Propagate()
        /// @return Output neurons values (result)
        unique_ptr<vector<double>> GetResult();

        /// @brief Print out result
        void PrintResult();
    };
}

#endif