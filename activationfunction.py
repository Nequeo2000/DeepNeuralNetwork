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
    def getGradientMatrix(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( Sigmoid.getStochasticGradient([[v]]) )

        gradientMarix = numpy.zeros(shape=(matrix.shape[1],matrix.shape[1]), dtype=float)
        numpy.fill_diagonal(gradientMarix,gradients)
        
        return gradientMarix
    
class Tanh:
    @staticmethod
    def calc(matrix: numpy.matrix):
        return numpy.tanh(matrix)
    
    @staticmethod
    def getStochasticGradient(matrix: numpy.matrix):
        return 1-math.pow(math.tanh(matrix[0][0]),2)
    
    @staticmethod
    def getGradientMatrix(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( Tanh.getStochasticGradient([[v]]) )

        gradientMarix = numpy.zeros(shape=(matrix.shape[1],matrix.shape[1]), dtype=float)
        numpy.fill_diagonal(gradientMarix,gradients)
        
        return gradientMarix
    
class ReLu:
    @staticmethod
    def calc(matrix: numpy.matrix):
        return numpy.maximum(matrix,0)
    
    @staticmethod
    def getStochasticGradient(matrix: numpy.matrix):
        return 0 if matrix[0][0]<0 else matrix[0][0]
    
    @staticmethod
    def getGradientMatrix(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( ReLu.getStochasticGradient([[v]]) )

        gradientMarix = numpy.zeros(shape=(matrix.shape[1],matrix.shape[1]), dtype=float)
        numpy.fill_diagonal(gradientMarix,gradients)
        
        return gradientMarix
    
class LeakyReLu:
    alpha = 0.1
    @staticmethod
    def setAlpha(alpha: float):
        LeakyReLu.alpha = alpha

    @staticmethod
    def calc(matrix: numpy.matrix):
        return numpy.where(matrix>0, matrix, matrix*LeakyReLu.alpha)
    
    @staticmethod
    def getStochasticGradient(matrix: numpy.matrix):
        return LeakyReLu.alpha if matrix[0][0]<0 else matrix[0][0]
    
    @staticmethod
    def getGradientMatrix(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( LeakyReLu.getStochasticGradient([[v]]) )

        gradientMarix = numpy.zeros(shape=(matrix.shape[1],matrix.shape[1]), dtype=float)
        numpy.fill_diagonal(gradientMarix,gradients)
        
        return gradientMarix

class Softmax:
    @staticmethod
    def calc(matrix: numpy.matrix):
        return scipy.special.softmax(matrix,axis=1)
    
    @staticmethod
    def getStochasticGradient(matrix: numpy.matrix):
        return 1 # will look into derivative of softmax another time
    
    @staticmethod
    def getGradientMatrix(matrix: numpy.matrix):
        gradients = []
        for v in matrix[0]:
            gradients.append( Softmax.getStochasticGradient([[v]]) )

        gradientMarix = numpy.zeros(shape=(matrix.shape[1],matrix.shape[1]), dtype=float)
        numpy.fill_diagonal(gradientMarix,gradients)
        
        return gradientMarix

def getActivationFunctions():
    return [Sigmoid,Tanh,ReLu,LeakyReLu,Softmax]