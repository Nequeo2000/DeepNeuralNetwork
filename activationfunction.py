import scipy.special
import numpy
import math

class Sigmoid:
    @staticmethod
    def calc(matrix):
        return scipy.special.expit(matrix)
    
    @staticmethod
    def getStochasticGradient(matrix):
        return matrix[0][0]*(1-matrix[0][0])
    
class Tanh:
    @staticmethod
    def calc(matrix):
        return numpy.tanh(matrix)
    
    @staticmethod
    def getStochasticGradient(matrix):
        return 1-math.pow(math.tanh(matrix[0][0]),2)
    
class ReLu:
    @staticmethod
    def calc(matrix):
        return numpy.maximum(matrix,0)
    
    @staticmethod
    def getStochasticGradient(matrix):
        return 0 if matrix[0][0]<0 else matrix[0][0]
    
class LeakyReLu:
    alpha = 0.1
    @staticmethod
    def setAlpha(alpha):
        LeakyReLu.alpha = alpha

    @staticmethod
    def calc(matrix):
        return numpy.where(matrix>0, matrix, matrix*LeakyReLu.alpha)
    
    @staticmethod
    def getStochasticGradient(matrix):
        return LeakyReLu.alpha if matrix[0][0]<0 else matrix[0][0]

class Softmax:
    @staticmethod
    def calc(matrix):
        return scipy.special.softmax(matrix,axis=1)
    
    @staticmethod
    def getStochasticGradient(matrix):
        return 1 # will look into derivative of softmax another time

def getActivationFunctions():
    return [Sigmoid,Tanh,ReLu,LeakyReLu,Softmax]