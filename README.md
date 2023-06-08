# DeepNeuralNetwork

A simple python implementation of a deep neural network.

## Optimzation Algorithms
- [stochastic gradient decent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

## Activation Functions
- [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)
- [tanh](https://en.wikipedia.org/wiki/Hyperbolic_functions)
- [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Leaky Relu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [softmax](https://en.wikipedia.org/wiki/Softmax_function)

## Example Code
```Python
from DeepNeuralNetwork.deepneuralnetwork import DeepNeuralNetwork
from DeepNeuralNetwork import activationfunction

nn = DeepNeuralNetwork(nodes=[2,5,3,2],
                        learningrate=0.1,
                        activations=[activationfunction.Sigmoid,
                                     activationfunction.ReLu,
                                     activationfunction.Softmax])

# training the network
nn.fit([0,1],[1,0])

# get prediction from network
print( nn.predict([0,1]) )
```
### Example Output
```Shell
[0.86699622 0.13300378]
```
