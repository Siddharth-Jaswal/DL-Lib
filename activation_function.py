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
