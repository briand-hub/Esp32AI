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

#ifndef BRIAND_NN_H
#define BRIAND_NN_H

#include "BriandInclude.hxx"

using namespace std;

namespace Briand {

    // Early declaration of Synapsis class needed in Neuron.
    class Synapsis;

    // Define an activation function type as a pointer to function that returns a double
    // by taking input with

    /// @brief Typedef (alias with C++ using) an activation function as a function returning a double and asking a const double& as parameter
    using ActivationFunction =  double (*)(const double&);

    /// @brief Typedef (alias with C++ using) an error calculation function as a function returning a double and asking two const double& as parameters (TARGET and OUTPUT)
    using ErrorFunction =  double (*)(const double&, const double&);

    /** @brief A neuron */
    class Neuron {
        public:

        /** @brief This neuron receives input from other neurons through Synapsis. 
         * If this is an input or bias neuron, then there is no connection.
         */
        unique_ptr<vector<unique_ptr<Synapsis>>> Inputs;

        /** @brief The output value of this neuron. 
         * If this is an input neuron, then this value is the input value.
         * Otherwise this is the calculated value (activated weighted sum of connected inputs)
        */
        double Value;

        /// @brief Neuron ID (not required)
        long Id;

        /// @brief Default constructor
        Neuron();
        
        /// @brief Constructor with initial value
        /// @param value Assigned initial value
        Neuron(const double& value);

        /// @brief Connect this neuron to other with given weight (other neuron will have one more input Synapsis)
        /// @param other The other neuron
        /// @param weight Synapsis weight. Default is 1.0
        void ConnectTo(const unique_ptr<Neuron>& other, double weight = 1.0);

        /// @brief Update the value, recalculating weighted sum from inputs with the given activation function.
        /// @param activationFunction Pointer to activation function to be called for calculations
        void UpdateValue(ActivationFunction);
    };

    /** @brief A weighted connection between two neurons */
    class Synapsis {
        public:

        /** @brief Source neuron pointer */
        Neuron* Source;

        /** @brief Connection weight. If this is an input, Weight must be always 1. */
        double Weight;
    };

    /** @brief The NN layer type (input, hidden, output ...) */
    enum class LayerType { Input, Hidden, Output, Kernel, Pooling };

    /** @brief A layer of neurons */
    class NeuralLayer {
        protected:

        /// @brief Pointer to activation function for this layer
        ActivationFunction _activationFunction;

        public:

        /// @brief Layer type (input, hidden, ...)
        LayerType Type;

        /// @brief Layer neurons
        unique_ptr<vector<unique_ptr<Neuron>>> Neurons;

        /// @brief Constructor with just LayerType. All must be done manually, no neurons.
        /// @param type Required, layer type
        /// @param activationFunction Required, activation function for the layer
        NeuralLayer(const LayerType& type, ActivationFunction);

        /// @brief Updates all layer's neurons values
        void UpdateNeurons();
    }; 

    /// @brief An empty Neural Network, without layers, neurons and connections.
    /// Has no particular methods, just basic data structure and propagation forward.
    /// Use it when you know what you are doing!
    class NeuralNetwork {
        protected:

        public:

        /// @brief There is always an input
        unique_ptr<NeuralLayer> InputLayer;

        /// @brief There might be some hidden layers
        unique_ptr<vector<unique_ptr<NeuralLayer>>> HiddenLayers;

        /// @brief There is always an output layer
        unique_ptr<NeuralLayer> OutputLayer;

        /// @brief Forward Propagation
        virtual void PropagateForward();

        /// @brief Constructor
        NeuralNetwork();
    };

    /// @brief Perceptron (one input layer, one output layer with single out, no hidden layers)
    class Perceptron : public NeuralNetwork {
        protected:

        public:

        /// @brief Create Perceptron with specified inputs. All inputs will be connected to output automatically.
        /// @param inputs Number of input neurons
        /// @param activationFunction Activation function to be used (pointer)
        Perceptron(const int& inputs, ActivationFunction activationFunction);

        virtual void PropagateForward() override;

        /// @brief Propagate backward
        /// @param error The error obtained, to be minimized
        virtual void PropagateBackward(const double& error);

        /// @brief Do a training session (forward and backward then backward)
        /// @param inputValues Input values
        /// @param target Expected output
        /// @param errorFunction Math function f(target, output) to use in order to calculate error for backpropagation
        /// @param error Save error (output parameter)
        virtual void Train(const unique_ptr<vector<double>>& inputValues, const double& target, ErrorFunction errorFunction, double& error);

        /// @brief Result from the single output (prediction). Method sets inputs, propagates forward and returns the value of output neuron.
        virtual double Predict(const unique_ptr<vector<double>>& inputValues);
    };

    /// @brief Fully connected Neural Network
    class FCNN : public NeuralNetwork {
        protected:

        public:

        /// @brief Set (or reset) the input layer
        /// @param inputs All input values
        virtual void SetInputLayer(const unique_ptr<double>& input);

        /// @brief Set (or reset) the number of output neurons
        /// @param outputs Number of output neurons
        virtual void SetOutputLayer(const int& outputs);

        /// @brief Adds an hidden layer
        virtual void AddHiddenLayer();

        /// @brief Connect all the network and propagate. Results can be obtained with
        virtual void PropagateForward();
        
        /// @brief Propagate backward
        /// @param error The error obtained, to be minimized
        virtual void PropagateBackward(const double& error);
    };
}

#endif