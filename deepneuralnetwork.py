import numpy
import math
try:
    import DeepNeuralNetwork.activationfunction
except ImportError:
    import activationfunction
try:
    import DeepNeuralNetwork.optimization
except ImportError:
    import optimizationfunction

class DeepNeuralNetwork:
    def __init__(self,nodes,learningrate=None,activations=None,optimization=None):
        self.lr = 1/math.sqrt(max(nodes)) if learningrate==None else learningrate
        self.weights = []
        for i in range(len(nodes)-1):
            layerWeights = numpy.random.uniform(-1,1,size=(nodes[i]+1,nodes[i+1]))
            self.weights.append(layerWeights)
        
        self.activationfunctions = (len(nodes)-2)*[activationfunction.Sigmoid]+[activationfunction.Softmax] if activations==None else activations
        self.optimization = optimizationfunction.StochasticGradientDecent if optimization == None else optimization

    def predict(self,input):
        input = optimizationfunction.addBias(input)
        
        output = numpy.matmul(input,self.weights[0])
        output = self.activationfunctions[0].calc(output)
        output = optimizationfunction.addBias(output)
        for i in range(1,len(self.weights)):            
            output = numpy.matmul(output,self.weights[i])
            output = self.activationfunctions[i].calc(output)
            output = optimizationfunction.addBias(output) if i < len(self.weights)-1 else output

        return output.flatten()
    
    def fit(self,input, expectedOutput):
        self.optimization(NN=self, input=input, expectedOutput=expectedOutput)
