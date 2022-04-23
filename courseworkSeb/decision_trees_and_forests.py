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
from sklearn import ensemble
from sklearn import tree




##############################################

# Split & Shuffle Dataset
scaled_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/scaled_dataset.csv')
clean_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/clean_dataset.csv')
clean_data = clean_data.astype('float')

train, test = train_test_split(scaled_data, test_size=0.2) #using 20% of our data.

train_target = train['Failed_Yes']
test_target = test['Failed_Yes']
train = train.drop('Failed_Yes', 1)
test = test.drop('Failed_Yes', 1)




##############################################
# Decision trees
decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(train, train_target)
plt.figure(figsize=(15, 15))
tree.plot_tree(decision_tree)


##############################################
# Random forests

random_forest = ensemble.RandomForestClassifier()
random_forest = random_forest.fit(train, train_target)



#############################################
# Evaluation metrics
# https://scikit-learn.org/stable/modules/model_evaluation.html

score = decision_tree.score(test, test_target)
predictions = decision_tree.predict(test)
print("The accuracy is: " + str(score*100) + "%")

tree_confusion_matrix = confusion_matrix(test_target, predictions)
# print(tree_confusion_matrix)
plt.figure(figsize=(9, 9))
sns.heatmap(tree_confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)

plt.figure()
tree_accuracy = accuracy_score(test_target, predictions)
print("The accuracy is given as: " + str(tree_accuracy*100)+"%")

tree_precision = precision_score(test_target, predictions)
print("The precision is given as: " + str(tree_precision*100)+"%")

tree_recall = recall_score(test_target, predictions)
print("The recall is given as: " + str(tree_recall*100)+"%")




plt.show()