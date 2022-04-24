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
from sklearn.linear_model import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *




######################################################################
# Split & Shuffle Dataset
scaled_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/scaled_dataset.csv')
clean_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/clean_dataset.csv')
clean_data = clean_data.astype('float')

train, test = train_test_split(scaled_data, test_size=0.2) #using 20% of our data.

train_target = train['Failed_Yes']
test_target = test['Failed_Yes']
train = train.drop('Failed_Yes', axis=1)
test = test.drop('Failed_Yes', axis=1)

# print(test_target)

######################################################################
# https://github.com/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb
# Regression Model
# Try mini-batch gradient descent.
# predict TORQUE from ROTATIONAL SPEED

y = train['Torque_Nm']
X = train['Rotational_speed_rpm']
y_test = test['Torque_Nm']
X_test = test['Rotational_speed_rpm']

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

sns.scatterplot(x=X, y=y, data=train)
lin_reg = LinearRegression()
# print(X, y)
# lesson: always turn into an array before fitting

sorted_indices = np.argsort(X)
sorted_indices_test = np.argsort(X_test)

X = np.array(X)
y = np.array(y)
X = np.reshape(X, (-1, 1))
y = np.reshape(y, (-1, 1))
X = X[sorted_indices]
y = y[sorted_indices]

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = np.reshape(X_test, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))
X_test = X_test[sorted_indices_test]
y_test = y_test[sorted_indices_test]

lin_reg.fit(X, y)
y_predict_lin_reg = lin_reg.predict(X)

plt.plot(X, y_predict_lin_reg, c="limegreen")

degree = 3
poly = PolynomialFeatures(degree=3, include_bias=False)

# print(X.size, y.size)
poly_features = poly.fit(X)
poly_features = poly.transform(X)
poly_regression = lin_reg.fit(poly_features, y)
y_predict_poly_reg = poly_regression.predict(poly_features)

plt.plot(X, y_predict_poly_reg, c="red", linewidth=3)
plt.legend(['Linear Regression', 'Polynomial Regression', 'Data'])
plt.title("Polynomial regression with degree "+str(degree))
percent_r2 = r2_score(y, y_predict_poly_reg)*100

######################################################################
# Cross Validation
# print(lin_reg.score(X_test, y_test))
"""
print(cross_val_score(poly_regression, X_test, y_test, cv=5, scoring='neg_mean_squared_error'))
print("The R2 score is : " + str(percent_r2) + "%")
print("The MSE for the polynomial regression: ", mean_squared_error(y, y_predict_poly_reg))
print("The MSE for the linear regression", mean_squared_error(y, y_predict_lin_reg))
print("The MAE for the polynomial regression :", mean_absolute_error(y, y_predict_poly_reg))# Note this is abs error
print("The explained variance score for the polynomial regression: ", explained_variance_score(y, y_predict_poly_reg))
print("The max error is for the poly regression is :", max_error(y, y_predict_poly_reg))
"""
# print(confusion_matrix(y, y_predict_poly_reg))
# print(poly_regression.score(X_test, y_test))


######################################################################
# Logistic Regression
#
# # Training set for logit
# sorted_indices_logit = np.argsort(train)
# logit_train = np.array(train)
# logit_train_target = np.array(train_target)
#
# logit_train = np.reshape(logit_train, (-1, 1))
# logit_train_target = np.reshape(logit_train_target, (-1, 1))
# logit_train = logit_train[sorted_indices_logit]
# logit_train_target = logit_train_target[sorted_indices_logit]
#

# # Testing set for logit
# sorted_indices_logit_test = np.argsort(test)
# logit_test = np.array(test)
# logit_test_target = np.array(test_target)
#
# logit_test = np.reshape(logit_test, (-1, 1))
# logit_test_target = np.reshape(logit_test_target, (-1, 1))
# logit_test = logit_test[sorted_indices_logit_test]
# logit_test_target = logit_test_target[sorted_indices_logit_test]

# we use a cv = 5 fold (cv generator used is stratified K-folds)
logit_regression = LogisticRegressionCV(cv=5, random_state=0, max_iter=500).fit(train, np.ravel(train_target))
# np.ravel()  ravel is used to flatten the array
score = logit_regression.score(test, test_target)
predictions = logit_regression.predict(test)
print("The accuracy is: " + str(score*100) + "%")

logit_confusion_matrix = confusion_matrix(test_target, predictions)
# print(logit_confusion_matrix)
plt.figure(figsize=(9, 9))
sns.heatmap(logit_confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)

plt.figure()
logit_accuracy = accuracy_score(test_target, predictions)
print("The accuracy is given as: " + str(logit_accuracy*100)+"%")

logit_precision = precision_score(test_target, predictions)
print("The precision is given as: " + str(logit_precision*100)+"%")

logit_recall = recall_score(test_target, predictions)
print("The recall is given as: " + str(logit_recall*100)+"%")





###
# Metrics to be used to measure logistic regression against:
# Confusion matrix
# accuracy
# precision




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
