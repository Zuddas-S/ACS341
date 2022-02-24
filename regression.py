import numpy as np
import matplotlib.pyplot as plt

no_data_points = 100
X = 2*np.random.rand(no_data_points, 1)
y = 3*X + 4 + np.random.rand(no_data_points, 1) # Defining the output according to y = 3x + 4 + noise equation.
X_b = np.append(np.ones((len(X), 1)), X, axis=1) # Adding the bias term.

# Defining the simple linear regression line
class Regression:

    def __init__(self, n_iter=1000, eta=0.01):
        self.theta = 0
        self.n_iter = n_iter
        self.eta = eta

    def initialize_weights(self):
        self.theta =  np.random.rand(2,1)*10

    def compute_loss(self, X, y):
        m = len(X)
        return 2.0/m * np.dot(X.T, (np.dot(X, self.theta) - y))

    def train(self, X, y):
        self.initialize_weights()
        self.error_iter = []
        self.theta_iter = []
        for i in range(self.n_iter):
            loss = self.compute_loss(X, y)
            self.theta = self.theta - self.eta*loss
            y_hat = np.dot(X, self.theta) # not an actual prediction but necessary for calculation of MSE.
            error = self.compute_mse(y_hat, y)
            self.error_iter.append(error)
            self.theta_iter.append(self.theta)
        return self

    def compute_mse(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

# Training the model.
theta = np.random.rand(2,1)*10
reg = Regression()
reg.train(X_b, y)
params = reg.theta_iter

# Visualising the regression line against data.
fig = plt.figure()
plt.scatter(X, y)
plt.plot(X, params[-1][0] + params[-1][1] * X, '-')
plt.show()


