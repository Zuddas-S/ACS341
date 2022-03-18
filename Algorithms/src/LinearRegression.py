from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt

no_data_points = 100
X = 2*np.random.rand(no_data_points, 1)
y = 3*X + 4 + np.random.rand(no_data_points, 1) # Defining the output according to y = 3x + 4 + noise equation.
X_b = np.append(np.ones((len(X), 1)), X, axis=1) # Adding the bias term.

# Defining the simple linear regression line
class LinearRegression:

    def __init__(self, n_iter=1000, eta=0.01):
        self.theta = 0
        self.n_iter = n_iter
        self.eta = eta

    def train(self, X, y):
        self._initialize_weights(X)
        self.error_iter = []
        self.theta_iter = []
        for i in range(self.n_iter):
            loss = self._compute_loss(X, y)
            self.theta = self.theta - self.eta*loss
            y_hat = np.dot(X, self.theta) # not an actual prediction but necessary for calculation of MSE.
            error = self._compute_mse(y_hat, y)
            self.error_iter.append(error)
            self.theta_iter.append(self.theta)
        return self
    
    def predict(self, X):
        return np.dot(X, self.theta)

    def _initialize_weights(self, X):
        n_outputs = X.shape[1]
        self.theta =  np.random.rand(n_outputs, 1)*10

    def _compute_loss(self, X, y):
        m = len(X)
        return 2.0/m * np.dot(X.T, (np.dot(X, self.theta) - y))

    def _compute_mse(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

# Training the model.
model = LinearRegression()
model.train(X_b, y)
y_pred = model.predict(np.array([3, 4]))
print(y_pred)
params = model.theta_iter

# Visualising the regression line against data.
fig = plt.figure()
plt.scatter(X, y)
plt.plot(X, params[-1][0] + params[-1][1] * X, '-')
plt.show()


