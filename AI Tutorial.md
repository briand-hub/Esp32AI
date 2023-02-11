# My AI notes and remarks

This file contains notes, remarks collected in years and years. I had to refresh some concepts and theory before doing this fun project. Hope this short tutorial will help you to understand AI and deep learning better. 

This is a learn-by-doing tutorial and every step of project is there documented and explained in detail, with math and theory.

## Basics - theory

Artificial Intelligence (AI) is a way to make a computer acting like a human. Of course, a program can always be written. For example I can write a program to do a simple sum *a+b* with just an instruction. However this is not intelligence because I am telling the computer what I want and the computer is doing that. Nothing else.

We have learned what a sum is when children, counting with fingers. Would it be possible a computer doing that like us?

Our brain is, essentialy, a sort of *elecrtonic processor*. Neurons are connected each other by *synapsis* and electric signals are sent between one and the other neuron through those connections. If a neuron is *activated* from incoming electric signals (mV, millivolts) then an electric signal is produced and propagated through the other connections. 

![](https://en.wikipedia.org/wiki/Biological_neuron_model#/media/File:Neuron3.png)
*Image from Wikipedia*

Such model can be easly converted in a program and the electric processes behind can be enough well translated into math. 

## Machine learning

One way to learn a computer how to solve problems is through Neural Networks (it is not the only way, but I think the most interesting one!).

Essentialy a Neural Network (NN) is a program that acts like neurons connected each other and **simulating** the human thinking/learning. In fact a neural network can *adjust* itself internal calculations in order to produce the expected results.

Given an input the network will try to produce an output. The output is just a series of calculations (computers are good in that!) that could be right or not.

Those results are then measured (compared to an expected value) and error is calculated. If error is more than a desidered threshold, this information is *"sent back"* in the program in order to produce a better calculation/result.

Doing that for many, many times (**epochs**) will led to an enough precise result (error is lower each time).

Essentialy the computer, providing a good program, could reproduce our way of learning things. In fact we learn, for example, what a cat is and then we are able to say "*yes this is a cat*". Then we can learn what is a Siamese cat and distinguish Siamese from Russian Blue for example. But this is done step-by-step (in fact, if we would not be able to say what a cat is, would be very difficult to say if this is a Siamese).

Through a computer program we have to give, for example, a cat image as input. The output should be "yes" or "no" (is a cat or not classification). But we do not want to write each single line of code to recognize a cat or hash all world's cat pics to a big dictionary with yes/no answers. 

We'd like to write a particular algorithm that given any pic as input will produce yes/no as output and that this algorithm will **automatically** adjust itself in order to learn what a cat is. Then, when we are satistfacted, this **model** of calculations can be used with all images we like. We are not interested in telling the computer what a cat is but we aim to teach the computer how a cat can be recognized.

This is, essentialy, machine learning: computer will be able to learn and adjust its knowledge itself. This can be done with statistics, math, and many other ways. But the most interesting one is Neural Networks, algorithms capable to simulate human brain.

## What is a Neural Network (NN)

### Basics

Neural Network concept is creating a data structure with neurons connected each other. Neurons wait for our input and calculations inside the network will produce an output. 

Like explained before, a neuron is a "unit" connected to other neuros. Neuros can be active or not and its signals are sent as millivolts to other neuros. Other neurons take those input signals and they can activate or not. And so on... million and million times.

Translating that into informatics: we have a set of neurons (a **layer** of neurons). They take an input and have connections to other neurons (**synapsis**). Input is translated into electric signal (**weighted**) and basing on that value, through an **activation function** the neuron output is (sent) **propagated** to the other layers. 

The first layer is the **input layer** (where we define one or more input for the network), then one (or more) **hidden layers** of neurons (may be **fully connected**) elaborate inputs and produces an output for the next layer. The final layer is the **output layer** (that could be one or more).

Simple network:

![](https://miro.medium.com/max/640/0*Bqpea_57RtV-kJ6D.webp)
*image from https://miro.medium.com*

A neural network without any hidden layer (so only input and output) is called **Perceptron**
A neural network with more than one hidden layer is called **Deep Neural Network (DNN)** and, as the name states, is used in the **Deep Learning**.

Translating it into simple informatics, a neuron "value" (the circles) can be C++ *objects*. Each object is connected to others (*pointers*). There will be a "value" (*double*) calculated by a function (*activation function*) and this value will be sent through connections to the other neurons. Of course the sent value could not be the same in each connection and to obtain this we can add a factor called **weight** (another *double*).

In schematics:

![](https://miro.medium.com/max/640/0*87ax_yzYZp-OUkFL.webp])
*image from https://miro.medium.com*

So, this is easy to translate in C++, you can find it in the [header file](components/briand_ai/include/BriandNN.hxx)

Generally, if we have a neuron $O$ receiving inputs from $n$ neurons having value of $I$ its "value" (output of the connected neurons) is  given by the easy formula of weighted sum:

$$ O = w_{1} * I_{1} + w_{2} * I_{2} + w_{3} * I_{3} + ... + w_{n} * I_{n} $$

Where:

 - $i$ is the neuron $i$ of $n$
 - $w_{i}$ is the weight assigned to the connection $i$
 - $I_{i}$ is the "value" of the neuron $i$

But... if all weights or inputs in a certain moment are all zero? The sum will be zero. And this 0 will be **propagated** to all other, making always output equal to zero. How to avoid this? It's simple, we can add a **bias** in one, two, or all layers (excluding output layer). A bias is a "standalone" neuron who has no inputs but is fully connected to all other neurons in the layer he lies.


![](https://miro.medium.com/max/640/0*WZ3ejB1QgidBrUak.webp)
*image from https://miro.medium.com*

So, the $O$ value calculation changes to:

$$ O = w_{1} * I_{1} + w_{2} * I_{2} + w_{3} * I_{3} + ... + w_{n} * I_{n} + w_{b} * b $$ 

Where:

 - $i$ is the neuron $i$ of $n$
 - $w_{i}$ is the weight assigned to the connection $i$
 - $I_{i}$ is the "value" of the neuron $i$
 - $b$ is the bias value

Sometimes bias has no weight associated and it is a fixed value marked with just $b$.

### Activation

The output value of the weighted sum could not be good for calculations and neuron values could be very variable. Biologically, if the signal is higher than a mV (millivolts) threshold the neuron is meant to be activated (in informatics we have a binary 1) but if this not happens then neuron is not activated (in informatics we have a binary 0). So we need some way to "flattern" neuron values to what is easy to be recognized as "active" or "inactive", or better, have a good "step". This can be done with simple math functions like **sigmoid** function. Of course, many other can be used also for performances. 

So, neuron "value" calculated with the weighted sum must be the argument of the **activation function**. We can write as:

$$ Oact = f( O ) = f( w_{1} * I_{1} + w_{2} * I_{2} + w_{3} * I_{3} + ... + w_{n} * I_{n} + w_{b} * b ) $$

The type of the $f$ function is not a guess, there are some rules to follow for good NN programming.

In project there are the following activation functions (see [header source](components/briand_ai/include/BriandMath.hxx))

#### Sigmoid (Logistic curve)

$$ Sigmoid(x) = { 1 \over 1 + e^{-x} } $$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/400px-Logistic-curve.svg.png)
*image from Wikipedia*

Sigmoid is widely used in neural networks.

#### ReLU

ReLU (**Re**ctified **L**inear **U**nit) is a simple function used (mostly) in hidden layers neuron activation. It is a rectified value:

$$ ReLU(x) = MAX(0, x) $$

So, if $x \gt 0$ then output of $ReLU(x) = x$ otherwise $ReLU(x) = 0$.

#### Other

Other functions that you can often find in NN are: $tanh$ and $softmax$.

#### How to choose activation function

Generally, follow this rule:

 - ReLU should be avoided everywhere but in hidden layers (because it is "restrictive")
 - In hidden layers never use sigmoid because of the *vanishing gradients* problem
 
So in hidden layers ReLU will be used in project, in the others sigmoid.

### Forward propagation

Now that the data structure is ready, we need to understand how data is propagated from inputs to output (**forward propagation**) with the weighted sum. This is done in project by calculating, starting from the first hidden layer (or the output layer if no hidden layer is defined) the value of each Neuron by calling the ``UpdateValue`` method. 
This method simply reads the ``Inputs`` vector synapsis. Foreach input, multiplied by the Weight, the sum is saved and then the activation function applied. 

This is done starting from the frist layer from the left till the output layer in the right.

When output is ready the **error** can be calculated respect to the **target** output (expected output). For example if output is 1 but we would expect a value of 2 then the error can be calculated.

To calculate the error there are many formulas can be used, however the most used in NN is the Mean Squared Error (**MSE**). Formula (one output, one target) is:

$$ MSE = {1 \over 2} * ( Target - Output )^2 $$

(the $1 \over 2$ factor simplifies next calculation, just accept it as is by now).

### Deep Neural Networks

It seems a difficult term, but the concept is very very easy: while a Perceptron has no hidden layer, a Neural Network has 1 hidden layer. Where there is more than one hidden layer the network is called *Deep* Neural Network.

---
**&#x1F44B; Find it in the project!**</span>

Math functions: [BriandMath.hxx](components/briand_ai/include/BriandMath.hxx) and [BriandMath.cpp](components/briand_ai/BriandMath.cpp). Math functions can be used then in all NN sources, by defining pointer type:

```C++
using ActivationFunction =  double (*)(const double& x); // Pointer to an activation function
using ErrorFunction =  double (*)(const double& target, const double& output); // Pointer to an error calculation function
```

NN basic classes in [BriandNN.hxx](components/briand_ai/include/BriandNN.hxx) and [BriandNN.cpp](components/briand_ai/BriandNN.cpp)

```C++
class Neuron; // This is a neuron. It has its value and all inputs synapsis from other neurons in the network. A method of UpdateValue does the value calculation.

class Synapsis; // This is the neuron "link". It connects the Source neuron to the current neuron with a Weight.

class NeuralLayer; // This is an array of neurons. Each layer has a type (input, output, hidden...) and they share the activation function. UpdateNeurons calls UpdateValue for each network in the layer.

class NeuralNetwork; // This is a basic Neural Netowork structure. Every NN has 1 input layer, 0 or more hidden layers, 1 output layer. The PropagateForward method does the forward propagation

class Perceptron; // A perceptron NN. Has one input and output layer.
```

---

### Neural network types

Here I paste some very nice images about common neural network types. 

Not all are covered by the project, but some are very interesting and funny. Foreach NN type I will place there some of the common use.

You can find a (mostly) comprehensive chart [here](https://i.stack.imgur.com/0WL34.jpg), follows details for NN used in project *source: [towardsai.net](https://towardsai.net/p/machine-learning/main-types-of-neural-networks-and-its-applications-tutorial-734480d7ec8e)*.

#### Feed Forward (FF):

Applications:

 - Data Compression.
 - Pattern Recognition.
 - Computer Vision.
 - Sonar Target Recognition.
 - Speech Recognition.
 - Handwritten Characters Recognition.

![](https://cdn-images-1.medium.com/max/1200/0*fbbKIJ-fxDyCfnd2.png)

#### Deep Feed-forward (DFF):

Applications:

 - Data Compression.
 - Pattern Recognition.
 - Computer Vision.
 - ECG Noise Filtering.
 - Financial Prediction.

![](https://cdn-images-1.medium.com/max/1200/0*06pOOKAQd5KURDZi.png)

#### Deep Convolutional Network (DCN):

Applications:

 - Identify Faces, Street Signs, Tumors.
 - Image Recognition.
 - Video Analysis.
 - NLP.
 - Anomaly Detection.
 - Drug Discovery.
 - Checkers Game.
 - Time Series Forecasting.

![](https://cdn-images-1.medium.com/max/1200/0*2EK9jpWmpogvIGet.png)

#### Support Vector Machines (SVM):

Applications:

 - Face Detection.
 - Text Categorization.
 - Classification.
 - Bioinformatics.
 - Handwriting recognition.

![](https://cdn-images-1.medium.com/max/1200/0*w2fuNgfTLExx1jlL.png)

## Machine Learning with Neural Network

So, at this point you should have in mind the structure and architecture of a Neural Network, how inputs are given and how output is generated. However this is nothing else than a simple and easy calculation (even wrong most of time). You also may have in mind some questions like *how to choose right weights values?* or *which bias value should I set?*. Well answer is just this: **do not worry, your computer will find the right ones itself**. What!? How is this possible? 

This process is just a series of math equations done once, twice and so on for a long time of **epochs** in order to let the computer find by itself the right values. This process is called **training** and the core of the training and the **learning** process is the **backpropagation**.

### Backpropagation

We discussed about *forward* propagation where input layer values are driven through the network to the output layer. When we have our output we can calculate an error (remember, the $MSE$) given the expected target value.

This error tells us *how much* we are wrong. We would like to communicate to each neuron of the previous layer this error in order to adjust each synapsis weight to obtain a lower error. Then, should do the same for the previous layer again and again till reaching the input. After that all weights will be updated (we don't have to worry about neuron's value because they depend on weights). If we can translate this into math equations then we can "teach" to our NN how to *self-adjust* or *learn from error*.

Done that, a forward propagation with updated weights will give a more precise output (with a lower error). We can repeat the process ($epoch$ times) until the error is as low as zero. 

This method found its base on the **gradient descent** algorithm. The idea is very simple: we would like a costant lower error at each epoch, this can be done with the math: in fact, the *derivative* of any fuctions tells us when the function is getting higher or lower and when zero, it is a (local) minimum. So we can use that to find the lowest possible error. 

In the case of a single neuron output layer, given a target output $T$, the current output $O$, in order to calculate the error we use the $MSE$ formula:

$$ MSE = {1 \over 2} * ( T - O )^2 $$

Then we want to know *how much* of that error is related to each previous-layer synapsis in order to change the weight and get a lower error. We define also that the new weight $\overline w$ is given by this formula:

$$ \overline w = w - \eta * \displaystyle \frac{\partial MSE}{\partial w} $$

*Note: here we use MSE as error calculation but other methods could be used!*

*Note 2: here I used just one output neuron. If more output neuros, simply calculate error for each one and the total error as the sum of all errors. The steps following must always refer to the total error.*

Explained: the new weight will be equal to old weight less than the *magnitude* of the current weight on the error, multiplied by a factor of $\eta$ (the **learning rate**). In other words, we'd like to know how much the weight *affects* the total error in order to correct it to a lower rate. We will have more chanches to "guess" it faster if a factor of $\eta$ is well "guessed". 

The $\frac{\partial MSE}{\partial w}$ is the (partial) **derivative** of the error respect to the weight (or the *gradient* of $MSE$ respect to $w$).

#### Concepts - Backpropagation at output layer

The main "problem" in solving the equation is that the weight $w$ is not *directly* related to the error $E$ but the relationship can be found applying *chain rule* and other math tricks.

I'll choose to follow the example from [Matt Mazur](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) because I think is very well written and explains well all math behind for a beginner.

In following formulas I refer to:

 - $f$ as the activation function
 - $f^{'}$ as the derivative of the activation function
 - $X_{inet}$ As the "net value" of a neuron $X_{i}$ (so, just the weighted sum of its inputs)
 - $X_{iout}$ As the "out value" of a neuron $X_{i}$, or $f(X_{inet})$ (so the activation function applied to weighted sum)
 - $E$ as the total error, calculated as $MSE$.
 - $T$ as the target value.
 - $w_{i}$ as the weight

Back to us... we forward propagated and we have got the output $O_{out}$ with an error of $E$. Suppose a netowrk done in this way:

![](tutorial_imgs/NNback1.png)

*Note: of course consider all weights and so on, but for tutorial and demonstration purposes only w1, w3 and w3 are shown.*

Now we want to "go back" using gradient descent algorithm. Taking the output node we want to know how much error is driven from the $w_{3}$. So we write this as $\displaystyle \frac{\partial E}{\partial w_{3}}$. Now we notice that:

![](tutorial_imgs/NNback2.png)

So, in order to reach $w_{3}$ "from" $E$ (the solid green arrow) we have to do three steps (three green arrows, or the *chain rule*): 

![](tutorial_imgs/NNback3.png)

Translate the arrows into math we have to write:

$$\displaystyle \frac{\partial E}{\partial w_{3}} = \frac{\partial E}{\partial O_{out}} * \frac{\partial O_{out}}{\partial O_{net}} * \frac{\partial O_{net}}{\partial w_{3}} $$

Now we notice that calculating error as the MSE (multiplied by $1\over2$) the derivative of the error is always $(O_{out} - T)$. This is the first term of the equation. Then the second term is, simply, the derivative of the choosen activation function $f$, so we have $f^{'}(O_{out})$ as the second term. Tha latest is the derivative of a weighted sum respect to $w_{3}$ (thus, all constants => zero) and this leds to the ${\partial (k+ .. + H_{2out}w_{3} + ... +k)} = H_{2out}$. So:

$$\displaystyle \frac{\partial E}{\partial w_{3}} = (O_{out} - T)*f^{'}(O_{out})*H_{2out} $$

*Note: see derivative in next sections in order to understand $f^{'}$ calculation.*

Given that, update the weight with the final formula:

$$ \overline w_{3} = w_{3} - \eta * (O_{out} - T)*f^{'}(O_{out})*H_{2out} $$

#### Concepts - Backpropagation at hidden and input layers

With hidden layers the procedure does not change so much. In fact, we always have out error $E$ but now we want to know how much $w_{2}$ affected error. So:

![](tutorial_imgs/NNback4.png)

$$\displaystyle \frac{\partial E}{\partial w_{2}} = \frac{\partial E}{\partial H_{2out}} * \frac{\partial H_{2out}}{\partial H_{2net}} * \frac{\partial H_{2net}}{\partial w_{2}} $$

Rember that the second term is always the derivative of the layer's activation function (to simplify we just call it once a time $f^{'}$ even if it could be a different function!). Also remember that the third term is always the weighted sum and this leds to:

$$\displaystyle \frac{\partial E}{\partial w_{2}} = \frac{\partial E}{\partial H_{2out}} *  f^{'}(H_{2out}) * H_{1out} $$

Here the first term is intended to be how much error respect to the output of $H_2{2}$ and then we have to apply chain rule again as $H_{2out}$ depends on the *previous* (*next*, going forward) weighted sum from $O_{net}$. This is written as:

$$\displaystyle \frac{\partial E}{\partial H_{2out}} = \frac{\partial E}{\partial O_{net}} * \frac{\partial O_{net}}{\partial H_{2out}}$$

The second term is $w_{3}$ fo course, and the first term can be obtain, again (*yes I know...*) with chain rule:

$$\displaystyle \frac{\partial E}{\partial O_{net}} = \frac{\partial E}{\partial O_{out}} * \frac{\partial O_{out}}{\partial O_{net}}$$

Byt hey! We have these values from the previous calculations! Therefore:

$$\displaystyle \frac{\partial E}{\partial w_{2}} = ( (O_{out}-T)*f^{'}(O_{out}) ) * f^{'}(H_{2out}) * H_{1out} $$

Then:

$$ \overline w_{2} = w_{2} - \eta * (O_{out}-T)*f^{'}(O_{out}) * f^{'}(H_{2out}) * H_{1out}  $$

We can proceed with the latest layer, the input layer. However procedure does not change so much, the final equation would be "longer" and more complex. The main focus of this part is to understand how gradient descent is calculated and its concept.

However to generalize this iteration matrix model is easier and can cover any kind of NN with the same procedure. In fact we could have any number of inputs, hidden layers and neurons (in any mix), output neurons and bias neurons. In order to give a generalized algorithm matrix model comes easier.


#### General backpropagation

Ia general scenario we would have a NN like this:






#### Resources and further reading

You can find an example (very well written) [here](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) about gradient descent algorithm applied to backpropagation step-by-step with all maths.

Also [here](https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c) there is a good tutorial with matrix-style calculations.

### An operative example

### Most used functions and their derivatives

### My name is $\eta$: *learning rate*, for friends.

## What is a Convolutional Neural Network (CNN)

### Kernels

### Pooling

### FCN (Fully Connected Network)

### Image formats

## Conclusions

future technology, creative, human being, "answers", state of art, train data quality is everything!
