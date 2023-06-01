import numpy
import scipy.special
import math

class DeepNeuralNetwork:
    def __init__(self,nodes,learningrate=None):
        self.lr = 1/math.sqrt(max(nodes)) if learningrate==None else learningrate
        self.weights = []
        for i in range(len(nodes)-1):
            layerWeights = numpy.random.uniform(-1,1,size=(nodes[i],nodes[i+1]))
            self.weights.append(layerWeights)
        
        self.activationFunction = lambda x: scipy.special.expit(x)

    def predict(self,input):
        input = numpy.array(input,ndmin=2)
        
        output = numpy.matmul(input,self.weights[0])
        output = self.activationFunction(output)
        for i in range(1,len(self.weights)):            
            output = numpy.matmul(output,self.weights[i])
            output = self.activationFunction(output)

        return output.flatten()
    
    def fit(self,input, expectedOutput):
        input = numpy.array(input,ndmin=2)
        expectedOutput = numpy.array(expectedOutput, ndmin=2)

        # calculate output of each layer
        outputs = []
        output = numpy.matmul(input,self.weights[0])
        output = self.activationFunction(output)
        outputs.append(output)
        for i in range(1,len(self.weights)):
            output = numpy.matmul(output,self.weights[i])
            output = self.activationFunction(output)
            outputs.append(output)
        
        # calculate errors for each layer
        errors = []
        error = numpy.subtract(expectedOutput,output)
        errors.append(error)
        for i in range(len(outputs)-1,0,-1):
            error = numpy.matmul( error, self.weights[i].T )
            errors.append(error)
        errors.reverse()

        # apply errors to weights
        gradient = outputs[0][0][0]*(1-outputs[0][0][0])
        for i in range(len(self.weights)-1,0,-1):
            alpha = self.lr*errors[i]*gradient
            self.weights[i] = self.weights[i]+numpy.matmul(outputs[i-1].T,alpha)
        alpha = self.lr*errors[0]*gradient
        self.weights[0] = self.weights[0]+numpy.matmul(input.T,alpha)

    def setWeights(self,w1,w2):
        self.weights = [w1,w2]