import scipy.special
import numpy
import math

class Sigmoid:
    @staticmethod
    def calc(matrix: numpy.matrix):
        return scipy.special.expit(matrix)
    
    @staticmethod
    def getStochasticGradient(matrix: numpy.matrix):
        return matrix[0][0]*(1-matrix[0][0])
    
    @staticmethod
    def getGradientArray(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( Softmax.getStochasticGradient([[v]]) )

        return gradients
    
class Tanh:
    @staticmethod
    def calc(matrix: numpy.matrix):
        return numpy.tanh(matrix)
    
    @staticmethod
    def getStochasticGradient(matrix: numpy.matrix):
        return 1-math.pow(math.tanh(matrix[0][0]),2)
    
    @staticmethod
    def getGradientArray(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( Tanh.getStochasticGradient([[v]]) )

        return gradients
    
class ReLU:
    @staticmethod
    def calc(matrix: numpy.matrix):
        return numpy.maximum(matrix,0)
    
    @staticmethod
    def getStochasticGradient(matrix: numpy.matrix):
        return 0 if matrix[0][0]<0 else matrix[0][0]
    
    @staticmethod
    def getGradientArray(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( ReLU.getStochasticGradient([[v]]) )

        return gradients
    
class LeakyReLU:
    alpha = 0.1
    @staticmethod
    def setAlpha(alpha: float):
        LeakyReLU.alpha = alpha

    @staticmethod
    def calc(matrix: numpy.matrix):
        return numpy.where(matrix>0, matrix, matrix*LeakyReLU.alpha)
    
    @staticmethod
    def getStochasticGradient(matrix: numpy.matrix):
        return LeakyReLU.alpha if matrix[0][0]<0 else matrix[0][0]
    
    @staticmethod
    def getGradientArray(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( LeakyReLU.getStochasticGradient([[v]]) )

        return gradients

class Softmax:
    @staticmethod
    def calc(matrix: numpy.matrix):
        return scipy.special.softmax(matrix,axis=1)
    
    @staticmethod
    def getStochasticGradient(matrix: numpy.matrix):
        return 1 # will look into derivative of softmax another time
    
    @staticmethod
    def getGradientArray(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( Softmax.getStochasticGradient([[v]]) )

        return gradients

def getList():
    return [Sigmoid,Tanh,ReLU,LeakyReLU,Softmax]