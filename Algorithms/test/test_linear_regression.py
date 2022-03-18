import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../src')

from LinearRegression import LinearRegression

no_data_points = 100
X = 2*np.random.rand(no_data_points, 1)
y = 3*X + 4 + np.random.rand(no_data_points, 1) # Defining the output according to y = 3x + 4 + noise equation.
X_b = np.append(np.ones((len(X), 1)), X, axis=1) # Adding the bias term.

# Training the model.
model = LinearRegression()
model.train(X_b, y)
y_pred = model.predict(np.array([3, 4]))
params = model.theta_iter

# Visualising the regression line against data on a scatter plot.
fig = plt.figure()
plt.scatter(X, y)
plt.plot(X, params[-1][0] + params[-1][1] * X, '-')
plt.show()

