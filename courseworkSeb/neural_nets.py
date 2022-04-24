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
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from math import sqrt



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


######################################################################
# Split & Shuffle Dataset


mlpc = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlpc.fit(train, train_target)
predict_train_mlpc = mlpc.predict(train)
predict_test_mlpc = mlpc.predict(test)

print("Confusion matrix train mlpc: \n", confusion_matrix(train_target, predict_train_mlpc))
print("Classification report train mlpc: \n", classification_report(train_target, predict_train_mlpc))
print("Confusion matrix test mlpc: \n", confusion_matrix(test_target, predict_test_mlpc))
print("Classification report test mlpc: \n", classification_report(test_target, predict_test_mlpc))


mlpr = MLPRegressor(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlpr.fit(train, train_target)
predict_train_mlpr = mlpr.predict(train)
predict_test_mlpr = mlpr.predict(test)


print("mlpr MAE train: \n", mean_absolute_error(train_target, predict_train_mlpr))
print("mlpr MSE train: \n", mean_squared_error(train_target, predict_train_mlpr))
print("mlpr explained variance: \n", explained_variance_score(train_target, predict_train_mlpr))
print("mlpr R2 score: \n", r2_score(train_target, predict_train_mlpr))

#
# print("Confusion matrix test mlpr: \n", confusion_matrix(test_target, predict_test_mlpr))
# print("Classification report test mlpr: \n", classification_report(test_target, predict_test_mlpr))

