from regression import data_sorting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn import svm


scaled_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/scaled_dataset.csv')
clean_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/clean_dataset.csv')
scaled_data = scaled_data.astype('float')

train, train_target, test, test_target = data_sorting(scaled_data, 'Failed_Yes', 0.2, False)


######################################################
# Hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

# Run once to get hyperparams
# svm = svm.SVC()
#
# clf = (GridSearchCV(svm, param_grid, cv=10)).fit(test, test_target)
# print('Best fit \n', clf.best_params_)

svm_opt = svm.SVC(C=1000,
                  gamma=0.001,
                  kernel='rbf')

svm_opt.fit(test, test_target)

predict_test_svm_opt = svm_opt.predict(test)

svm_confusion_matrix_opt = confusion_matrix(test_target, predict_test_svm_opt)
class_rep_opt = classification_report(test_target, predict_test_svm_opt)
print("Classification report train mlpc_opt: \n", class_rep_opt)
print("Confusion matrix train mlpc_opt: \n", svm_confusion_matrix_opt)
# print(tree_confusion_matrix)


cmn = 100*svm_confusion_matrix_opt.astype('float') / svm_confusion_matrix_opt.sum(axis=1)[:, np.newaxis]  # normalise confusion matrix
f = plt.figure(figsize=(30, 25))
ax = sns.heatmap(cmn,
                 annot=True,
                 fmt=".3f",
                 linewidths=1,
                 vmin=0,
                 vmax=100,
                 square=True,
                 cmap='flare',
                 annot_kws={"size": 100 / np.sqrt(len(svm_confusion_matrix_opt))})

plt.ylabel('Actual label %', fontsize=45)
plt.xlabel('Predicted label %', fontsize=45)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=50)
plt.title('Support Vector Machine Confusion Matrix', fontsize=50)
f.savefig('/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/svm_conf_matrix.png')




