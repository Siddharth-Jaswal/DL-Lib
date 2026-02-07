import numpy as np 

class PERCEPTRON:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.n_iters = num_iterations
        self.activation_function = self._relu_function
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                z = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_function(z)

                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update
    
    def _relu_function(self,x):
        return np.maximum(0,x)
    
    def predict(self,x):
        z = np.dot(x, self.weights) + self.bias
        y_pred = self.activation_function(z)
        return y_pred