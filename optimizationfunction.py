import numpy
import math

def layerNormalization(nn, values: numpy.matrix):
    if not nn.layerNormalization:
        return values

    mean = numpy.mean(values)
    stdDeviation = numpy.std(values)
    values = (values-mean)/stdDeviation
    return values

def addBias(nn, array):
    if isinstance(array,numpy.ndarray):
        array = array.flatten().tolist()
    bias = 1 if nn.useBias else 0
    return numpy.array(array+[bias],ndmin=2)

def StochasticGradientDecent(NN, input, expectedOutput):
    input = addBias(NN,input)
    expectedOutput = numpy.array(expectedOutput, ndmin=2)

    # calculate output of each layer
    outputs = []
    output = numpy.matmul(input,NN.weights[0])
    output = layerNormalization(NN,output)
    output = NN.activationfunctions[0].calc(output)
    output = addBias(NN,output)
    outputs.append(output)
    for i in range(1,len(NN.weights)):
        output = numpy.matmul(output,NN.weights[i])
        if i < len(NN.weights)-1:
            output = layerNormalization(NN,output)
        output = NN.activationfunctions[i].calc(output)
        if i < len(NN.weights)-1:
            output = addBias(NN,output)
        outputs.append(output)
    
    # calculate errors for each layer
    errors = []
    error = numpy.subtract(expectedOutput,output)
    errors.append(error)
    for i in range(len(outputs)-1,-1,-1):
        # take out bias error for error calculation
        error = numpy.array( error.flatten().tolist()[0:NN.weights[i].shape[1]] ,ndmin=2)
        error = numpy.matmul( error, NN.weights[i].T )
        errors.append(error)
    errors.reverse()
    errors[0] = numpy.array( errors[0].flatten().tolist()[0:-1] ,ndmin=2)

    # apply errors to weights
    for i in range(len(NN.weights)-1,0,-1):
        gradient = NN.activationfunctions[i].getStochasticGradient(outputs[i])
        if i < len(NN.weights)-1:
            error = numpy.array( errors[i+1].flatten().tolist()[0:-1] ,ndmin=2)
        else: error = errors[i+1]
        alpha = NN.lr*error*gradient
        NN.weights[i] += numpy.matmul(outputs[i-1].T,alpha)
    gradient = NN.activationfunctions[0].getStochasticGradient(outputs[0])
    error = numpy.array( errors[0+1].flatten().tolist()[0:-1] ,ndmin=2)
    alpha = NN.lr*error*gradient
    NN.weights[0] += numpy.matmul(input.T,alpha)

    return errors

def GradientDecent(NN, input, expectedOutput):
    input = addBias(NN,input)
    expectedOutput = numpy.array(expectedOutput, ndmin=2)

    # calculate output of each layer
    outputs = []
    output = numpy.matmul(input,NN.weights[0])
    output = layerNormalization(NN,output)
    output = NN.activationfunctions[0].calc(output)
    output = addBias(NN,output)
    outputs.append(output)
    for i in range(1,len(NN.weights)):
        output = numpy.matmul(output,NN.weights[i])
        if i < len(NN.weights)-1:
            output = layerNormalization(NN,output)
        output = NN.activationfunctions[i].calc(output)
        if i < len(NN.weights)-1:
            output = addBias(NN,output)
        outputs.append(output)
    
    # calculate errors for each layer
    errors = []
    error = numpy.subtract(expectedOutput,output)
    errors.append(error)
    for i in range(len(outputs)-1,-1,-1):
        # take out bias error for error calculation
        error = numpy.array( error.flatten().tolist()[0:NN.weights[i].shape[1]] ,ndmin=2)
        error = numpy.matmul( error, NN.weights[i].T )
        errors.append(error)
    errors.reverse()
    errors[0] = numpy.array( errors[0].flatten().tolist()[0:-1] ,ndmin=2)

    # apply errors to weights
    for i in range(len(NN.weights)-1,0,-1):
        relevantOutput = outputs[i] if i == len(NN.weights)-1 else numpy.delete(outputs[i], outputs[i].shape[1]-1,1)
        gradient = NN.activationfunctions[i].getGradientMatrix( relevantOutput )
        if i < len(NN.weights)-1:
            error = numpy.array( errors[i+1].flatten().tolist()[0:-1] ,ndmin=2)
        else: error = errors[i+1]
        alpha = NN.lr*numpy.matmul(error,gradient)
        NN.weights[i] += numpy.matmul(outputs[i-1].T,alpha)
    gradient = NN.activationfunctions[0].getGradientMatrix( numpy.array(outputs[0][0][0:-1], ndmin=2) )
    error = numpy.array( errors[0+1].flatten().tolist()[0:-1] ,ndmin=2)
    alpha = NN.lr*numpy.matmul(error,gradient)
    NN.weights[0] += numpy.matmul(input.T,alpha)

    return errors