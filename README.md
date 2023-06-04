# DeepNeuralNetwork

A simple python implementation of a deep neural network.

## Optimzation Algorithms
- [stochastic gradient decent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

## Activation Functions
- [Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function)

## Example Code
```Python
nn = DeepNeuralNetwork(nodes=[2,5,3,1], learningrate=0.1)

# training the network
nn.fit([0,1],[1])

# get prediction from network
print( nn.predict([0,1]) )
```
### Example Output
```Shell
[0.5542525]
```
