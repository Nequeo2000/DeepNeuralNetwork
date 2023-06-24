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
                 learningrate = None,
                 layerNormalization = False, 
                 activations = None, 
                 optimization = None,):
        self.lr = 1/math.sqrt(max(nodes)) if learningrate==None else learningrate
        self.weights = []
        for i in range(len(nodes)-1):
            layerWeights = numpy.random.uniform(-1,1,size=(nodes[i]+1,nodes[i+1]))
            self.weights.append(layerWeights)
        
        self.activationfunctions = (len(nodes)-2)*[activationfunction.Sigmoid]+[activationfunction.Softmax] if activations==None else activations
        self.optimization = optimizationfunction.GradientDecent if optimization == None else optimization
        self.layerNormalization = layerNormalization

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
        if self.layerNormalization:
            output = optimizationfunction.layerNormalization(output)
        output = self.activationfunctions[0].calc(output)
        output = optimizationfunction.addBias(output)
        for i in range(1,len(self.weights)):            
            output = numpy.matmul(output,self.weights[i])
            if self.layerNormalization and i < len(self.weights)-1:
                output = optimizationfunction.layerNormalization(output)
            output = self.activationfunctions[i].calc(output)
            output = optimizationfunction.addBias(output) if i < len(self.weights)-1 else output

        return output.flatten()
    
    def fit(self, input: "list[float]", expectedOutput: "list[float]") -> "list[list[float]]":
        return self.optimization(NN=self, input=input, expectedOutput=expectedOutput)

class GenerativeAdverserialNetwork:
    def __init__(self, generatorNodes: "list[int]", 
                 discriminatorNodes: "list[int]",
                 learningrates=None, 
                 activations=None):
        self.generator = DeepNeuralNetwork(nodes=generatorNodes,
                                           learningrate=learningrates[0] if learningrates != None else None,
                                           layerNormalization=True,
                                           activations=(len(generatorNodes)-2)*[activationfunction.Tanh]+[activationfunction.Sigmoid] if activations == None else activations[0])
        
        self.discriminator = DeepNeuralNetwork(discriminatorNodes, 
                                               learningrate=learningrates[1] if learningrates != None else None,
                                               layerNormalization=True,
                                               activations=None if activations == None else activations[1])
        self.generator.setLearningrate( self.generator.getLearningrate()*-1 )

    def generate(self, input: "list[float]"):
        return self.generator.predict(input)
    
    def fit(self, input: "list[float]"):
        generatorInput = numpy.random.uniform(-1,1,size=(1,self.generator.weights[0].shape[0]-1))

        self.discriminator.fit(input=input, expectedOutput=[1,0])
        errors = self.discriminator.fit(input=self.generate( generatorInput ), expectedOutput=[0,1])
        self.generator.fit(input=generatorInput, expectedOutput=input+errors[0])