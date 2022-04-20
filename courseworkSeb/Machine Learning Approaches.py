"""
Header



"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression



######################################################################
# Split & Shuffle Dataset
scaled_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/scaled_dataset.csv')
#print(scaled_data.head())

#target = scaled_data['Failed_Yes']
#scaled_data = scaled_data.drop('Failed_Yes')
#print(target.head())
train, test = train_test_split(scaled_data, test_size=0.2)
train_target = train['Failed_Yes'].astype('float')
test_target = test['Failed_Yes'].astype('float')
#print(test_target)


######################################################################
# https://github.com/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb
# Regression Model
# Try mini-batch gradient descent.
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