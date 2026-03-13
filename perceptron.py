class ClassicalPerceptron:

    def __init__(self, X, y, lr):
        self.samples, self.features = len(X), len(X[0])
        self.y = y
        self.lr = lr
        self.X = X
        self.w = [0] * (self.features + 1) 

    def fit(self, epochs):

        for _ in range(epochs):

            early_stopping = True

            for i in range(self.samples):

                zi = self.w[0]

                for j in range(self.features):
                    zi += self.w[j+1] * self.X[i][j]

                y_predi = 1 if zi >= 0 else 0

                if y_predi != self.y[i]:

                    early_stopping = False
                    error = self.y[i] - y_predi

                    self.w[0] += self.lr * error

                    for j in range(self.features):
                        self.w[j+1] += self.lr * error * self.X[i][j]

            if early_stopping:
                break
    
    def predict(self, X):
        y_preds = []
        for i in range(len(X)):
            zi = self.w[0]
            for j in range(self.features):
                zi += self.w[j+1] * X[i][j]
            y_preds.append(1 if zi >= 0 else 0)
        return y_preds
    

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]


import random

def generate_dataset(samples):

    w = [2, -1, 3, 0.5, -2]
    b = 0.7

    X = []
    y = []

    for _ in range(samples):

        x = [random.uniform(-5,5) for _ in range(5)]

        z = b
        for j in range(5):
            z += w[j] * x[j]

        label = 1 if z >= 0 else 0

        X.append(x)
        y.append(label)

    return X, y



X_train, y_train = generate_dataset(10000)
X_test, y_test = generate_dataset(200)



perceptron = ClassicalPerceptron(X_train, y_train, lr=0.01)
perceptron.fit(epochs=1000)


preds = perceptron.predict(X_test)

correct = 0
for i in range(len(y_test)):
    if preds[i] == y_test[i]:
        correct += 1

accuracy = correct / len(y_test)

print("Accuracy:", accuracy)
