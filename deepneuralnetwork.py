import numpy
import scipy.special
import math

class DeepNeuralNetwork:
    def __init__(self,nodes,learningrate=None):
        self.lr = 1/math.sqrt(max(nodes)) if learningrate==None else learningrate
        self.weights = []
        for i in range(len(nodes)-1):
            layerWeights = numpy.random.uniform(-1,1,size=(nodes[i]+1,nodes[i+1]))
            self.weights.append(layerWeights)
        
        self.activationFunction = lambda x: scipy.special.expit(x)

    @staticmethod
    def __addBias(array):
        if isinstance(array,numpy.ndarray):
            array = array.flatten().tolist()
        array.append(1)

        return numpy.array(array,ndmin=2)

    def predict(self,input):
        input = self.__addBias(input)
        
        output = numpy.matmul(input,self.weights[0])
        output = self.activationFunction(output)
        output = self.__addBias(output)
        for i in range(1,len(self.weights)):            
            output = numpy.matmul(output,self.weights[i])
            output = self.activationFunction(output)
            output = self.__addBias(output) if i < len(self.weights)-1 else output

        return output.flatten()
    
    def fit(self,input, expectedOutput):
        input = self.__addBias(input)
        expectedOutput = numpy.array(expectedOutput, ndmin=2)

        # calculate output of each layer
        outputs = []
        output = numpy.matmul(input,self.weights[0])
        output = self.activationFunction(output)
        output = self.__addBias(output)
        outputs.append(output)
        for i in range(1,len(self.weights)):
            output = numpy.matmul(output,self.weights[i])
            output = self.activationFunction(output)
            output = self.__addBias(output) if i < len(self.weights)-1 else output
            outputs.append(output)
        
        # calculate errors for each layer
        errors = []
        error = numpy.subtract(expectedOutput,output)
        errors.append(error)
        for i in range(len(outputs)-1,0,-1):
            # take out bias error for error calculation
            error = numpy.array( error.flatten().tolist()[0:self.weights[i].shape[1]] ,ndmin=2)
            error = numpy.matmul( error, self.weights[i].T )
            errors.append(error)
        errors.reverse()

        # apply errors to weights
        gradient = outputs[0][0][0]*(1-outputs[0][0][0])
        for i in range(len(self.weights)-1,0,-1):
            if i < len(self.weights)-1:
                error = numpy.array( errors[i].flatten().tolist()[0:-1] ,ndmin=2)
            else: error = errors[i]
            alpha = self.lr*error*gradient
            self.weights[i] = self.weights[i]+numpy.matmul(outputs[i-1].T,alpha)
        error = numpy.array( errors[0].flatten().tolist()[0:-1] ,ndmin=2)
        alpha = self.lr*error*gradient
        self.weights[0] = self.weights[0]+numpy.matmul(input.T,alpha)
