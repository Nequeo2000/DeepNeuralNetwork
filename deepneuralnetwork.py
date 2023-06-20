import numpy
import math
try:
    import DeepNeuralNetwork.activationfunction as activationfunction
except ImportError:
    import activationfunction
try:
    import DeepNeuralNetwork.optimizationfunction as optimizationfunction
except ImportError:
    import optimizationfunction

class DeepNeuralNetwork:
    def __init__(self, nodes: "list[int]", 
                 learningrate: float, 
                 activations: list, 
                 optimization: list):
        self.lr = 1/math.sqrt(max(nodes)) if learningrate==None else learningrate
        self.weights = []
        for i in range(len(nodes)-1):
            layerWeights = numpy.random.uniform(-1,1,size=(nodes[i]+1,nodes[i+1]))
            self.weights.append(layerWeights)
        
        self.activationfunctions = (len(nodes)-2)*[activationfunction.Sigmoid]+[activationfunction.Softmax] if activations==None else activations
        self.optimization = optimizationfunction.GradientDecent if optimization == None else optimization

    def setLearningrate(self, lr:float):
        self.lr = lr

    def getLearningrate(self) -> float:
        return self.lr
    
    def setWeights(self, weights: "list[list[float]]"):
        self.weights = weights

    def getWeights(self) -> "list[list[float]]":
        return self.weights

    def predict(self, input: "list[float]") -> "list[float]":
        input = optimizationfunction.addBias(input)
        
        output = numpy.matmul(input,self.weights[0])
        output = self.activationfunctions[0].calc(output)
        output = optimizationfunction.addBias(output)
        for i in range(1,len(self.weights)):            
            output = numpy.matmul(output,self.weights[i])
            output = self.activationfunctions[i].calc(output)
            output = optimizationfunction.addBias(output) if i < len(self.weights)-1 else output

        return output.flatten()
    
    def fit(self, input: "list[float]", expectedOutput: "list[float]") -> "list[list[float]]":
        return self.optimization(NN=self, input=input, expectedOutput=expectedOutput)