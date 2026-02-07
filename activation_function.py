import numpy as np

# x will be a 1-d numpy array

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    t = np.tanh(x)
    return 1 - t**2

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return (x > 0).astype(float)

def leaky_ReLU(x,a=0.01):
    arr = []
    for xi in x:
        if xi > 0:
            arr.append(xi)
        else:
            arr.append(a*xi)
    return np.array(arr)

def dleaky_ReLU(x, a=0.01):
    arr = []
    for xi in x:
        if xi > 0:
            arr.append(1)
        else:
            arr.append(a)
    return np.array(arr)