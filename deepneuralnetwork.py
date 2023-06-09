import numpy
import math
import json
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
                 useBias = False,
                 layerNormalization = False, 
                 activations = None, 
                 optimization = None,):
        self.lr = 1/math.sqrt(max(nodes)) if learningrate==None else learningrate
        self.weights = []
        for i in range(len(nodes)-1):
            layerWeights = numpy.random.uniform(-1,1,size=(nodes[i]+1,nodes[i+1]))
            self.weights.append(layerWeights)
        
        self.useBias = useBias
        self.layerNormalization = layerNormalization
        
        self.activationfunctions = (len(nodes)-2)*[activationfunction.Sigmoid]+[activationfunction.Softmax] if activations==None else activations
        self.optimization = optimizationfunction.GradientDecent if optimization == None else optimization

    @staticmethod
    def load(filename: str) -> "DeepNeuralNetwork":
        obj = json.loads(open(filename, "r").read())
        nn = DeepNeuralNetwork([1,1])
        
        nn.lr = obj["lr"]
        nn.useBias = obj["useBias"]
        nn.layerNormalization = obj["layerNorm"]
        nn.optimization = optimizationfunction.getList()[obj["optimization"]]

        nn.activationfunctions = []
        for i in range(len(obj["activations"])):
            nn.activationfunctions.append(activationfunction.getList()[obj["activations"][i]])

        nn.weights = []
        for i in range(0,len(obj["weights"]),2):
            shape = obj["weights"][i]
            weights = obj["weights"][i+1]
            weights = numpy.array(weights)
            weights.shape = shape
            nn.weights.append(weights)

        return nn

    def __toJSON(self):
        class Object(object):
            pass
        obj = Object()

        obj.lr = self.lr
        obj.useBias = self.useBias
        obj.layerNorm = self.layerNormalization
        obj.optimization = optimizationfunction.getList().index(self.optimization)

        obj.activations = []
        for i in range(len(self.activationfunctions)):
            obj.activations.append(activationfunction.getList().index(self.activationfunctions[i]))

        obj.weights = []
        for i in range(len(self.weights)):
            obj.weights.append(self.weights[i].shape)
            obj.weights.append(self.weights[i].flatten().tolist())

        return json.dumps(obj=obj.__dict__)

    def setLearningrate(self, lr:float):
        self.lr = lr

    def getLearningrate(self) -> float:
        return self.lr
    
    def setWeights(self, weights: "list[list[float]]"):
        self.weights = weights

    def getWeights(self) -> "list[list[float]]":
        return self.weights

    def toFile(self, filename: str):
        open(filename, "w").write(self.__toJSON())

    def predict(self, input: "list[float]") -> "list[float]":
        input = optimizationfunction.addBias(self,input)

        output = numpy.matmul(input,self.weights[0])
        output = optimizationfunction.layerNormalization(self,output)
        output = self.activationfunctions[0].calc(output)
        output = optimizationfunction.addBias(self,output)
        for i in range(1,len(self.weights)):            
            output = numpy.matmul(output,self.weights[i])
            if i < len(self.weights)-1:
                output = optimizationfunction.layerNormalization(self,output)
            output = self.activationfunctions[i].calc(output)
            output = optimizationfunction.addBias(self,output) if i < len(self.weights)-1 else output

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
                                           useBias=True,
                                           layerNormalization=True,
                                           optimization=optimizationfunction.GradientDecent,
                                           activations=(len(generatorNodes)-1)*[activationfunction.Tanh] if activations == None else activations[0])
        
        self.discriminator = DeepNeuralNetwork(discriminatorNodes, 
                                               learningrate=learningrates[1] if learningrates != None else None,
                                               useBias=True,
                                               layerNormalization=True,
                                               optimization=optimizationfunction.GradientDecent,
                                               activations=None if activations == None else activations[1])

    def generate(self, input: "list[float]"):
        return self.generator.predict(input)
    
    def fit(self, input: "list[float]"):
        self.fitDiscriminator(input=input)
        self.fitGenerator(input=input)

    def fitDiscriminator(self, input: "list[float]"):
        generatorInput = numpy.random.uniform(-1,1,size=(1,self.generator.weights[0].shape[0]-1))

        self.discriminator.fit(input=input, expectedOutput=[1,0])
        self.discriminator.fit(input=self.generate(generatorInput), expectedOutput=[0,1])

    def fitGenerator(self, input: "list[float]"):
        generatorInput = numpy.random.uniform(-1,1,size=(1,self.generator.weights[0].shape[0]-1))
        generatedOutput = self.generate(generatorInput)

        discLr = self.discriminator.getLearningrate()
        self.discriminator.setLearningrate(0)
        errors = self.discriminator.fit(input=generatedOutput, expectedOutput=[1,0])
        self.generator.fit(input=generatorInput, expectedOutput=generatedOutput+errors[0])
        self.discriminator.setLearningrate(discLr)