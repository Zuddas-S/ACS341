"""
Header



"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


######################################################################
# Split & Shuffle Dataset
scaled_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/scaled_dataset.csv')
clean_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/clean_dataset.csv')
clean_data = clean_data.astype('float')

train, test = train_test_split(scaled_data, test_size=0.2)
train_target = train['Failed_Yes']
test_target = test['Failed_Yes']


print(test_target)

######################################################################
# https://github.com/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb
# Regression Model
# Try mini-batch gradient descent.
# predict TORQUE from ROTATIONAL SPEED

y = train['Torque_Nm']
X = train['Rotational_speed_rpm']

##################################
# Regression setup

eta = 0.1
m = 100
theta_path_mgd = []
n_iter = 50
mini_batch_size = 20
np.random.seed(42)
theta = np.random.randn(2, 1)
t0, t1 = 200, 1000

##################################
# start with a simple linear regression

X_b = np.c_[np.ones((X.size, 1)), X] # Ensure the ones array that concatenates is the same size
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # standard eqn

X_new = np.array([[0], [2]]) # Whats the reason for the xnew?
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

print(y_predict)
sns.scatterplot(x=X, y=y, data=train)
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict_lin_reg = lin_reg.predict(X)
sns.lineplot(x=X, y=y_predict, data=train)
"""
Need to ensure dimensions line up with x and y of predicted


"""

print(lin_reg.intercept_, lin_reg.coef_)

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)


print(theta_best_svd)

for iteration in range(n_iter):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients



def learning_schedule(t):
    return t0/(t+t1)


t = 0

"""
for epoch in range(n_iter):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
"""
"""
lr = LinearRegression()
X = train[['Air_temperature_K']]
print(X.shape)
y = train_target
# lr.fit(X, y)

degree = 3
polyreg = make_pipeline(PolynomialFeatures(degree), lr)
polyreg.fit(X, y)
plt.figure()
plt.scatter(X, y)
plt.plot(X, polyreg.predict(X), color="black")
plt.title("Polynomial regression with degree "+str(degree))
"""

######################################################################
# Cross Validation

######################################################################
# Logistic Regression

######################################################################
# Decision Trees

###################################
# Random Forests

######################################################################
# Neural Nets

######################################################################
# SVM

######################################################################
# Show plots
plt.show()
