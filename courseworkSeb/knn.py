from regression import data_sorting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import *
from sklearn.linear_model import *
from sklearn.pipeline import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.neighbors import *


scaled_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/scaled_dataset.csv')
clean_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/clean_dataset.csv')
scaled_data = scaled_data.astype('float')

train, train_target, test, test_target = data_sorting(scaled_data, 'Failed_Yes', 0.2, False)

########################################################################
# Tuning hyperparams
leaf_size = list(range(1, 50))
n_neighbors = list(range(1, 30))
algorithm = list(['auto', 'ball_tree', 'kd_tree', 'brute'])
p = [1, 2]  # Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p, algorithm=algorithm)

knn = KNeighborsClassifier()
# clf = (GridSearchCV(knn, hyperparameters, cv=10)).fit(test, test_target)
# print('Best fit \n', clf.best_params_)


########################################################################
# Implementation
knn_opt = KNeighborsClassifier(algorithm='auto', leaf_size=1, n_neighbors=16, p=1)
knn_predict = knn_opt.fit(train, train_target)
knn_predict = knn_opt.predict(test)

print(classification_report(test_target, knn_predict))
print(confusion_matrix(test_target, knn_predict))



