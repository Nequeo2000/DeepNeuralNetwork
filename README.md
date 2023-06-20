# DeepNeuralNetwork

A simple python implementation of a deep neural network.

## Optimzation Algorithms
- [Stochastic Gradient Decent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- [Gradient Decent](https://en.wikipedia.org/wiki/Gradient_descent)

## Activation Functions
- [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)
- [Tanh](https://en.wikipedia.org/wiki/Hyperbolic_functions)
- [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Leaky Relu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Softmax](https://en.wikipedia.org/wiki/Softmax_function)

## Example Code
```Python
from DeepNeuralNetwork.deepneuralnetwork import DeepNeuralNetwork
from DeepNeuralNetwork import activationfunction
from DeepNeuralNetwork import optimizationfunction

nn = DeepNeuralNetwork(nodes=[2,5,3,2],
                        learningrate=0.1,
                        layerNormalization=True,
                        activations=[activationfunction.Sigmoid,
                                     activationfunction.ReLu,
                                     activationfunction.Softmax],
                        optimization=optimizationfunction.GradientDecent)

# training the network
nn.fit([0,1],[1,0])

# get prediction from network
print( nn.predict([0,1]) )
```
### Example Output
```Shell
[0.86699622 0.13300378]
```
