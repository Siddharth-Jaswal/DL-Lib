import numpy as np

class PERCEPTRON:

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for i in range(n_samples):
                z = np.dot(X[i], self.w) + self.b
                y_pred = 1 if z >= 0 else -1

                if y[i] != y_pred:
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return np.where(z >= 0, 1, -1)
