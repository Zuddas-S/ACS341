"""
Header



"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import *



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
# Tuning

mlp_gs = MLPClassifier(max_iter=10000)

parameter_space = {
    'hidden_layer_sizes': [(20,), (7,), (7, 1, 5), (7, 5, 1)],
    'activation': ['tanh', 'relu', 'identity', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

# clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
# clf.fit(train, train_target)  # X is train samples and y is the corresponding labels

"""
Best params: 
  {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (20,), 'learning_rate': 'constant', 'solver': 'lbfgs'}
"""

# print('Best params: \n', clf.best_params_)


######################################################################
#

mlpc = MLPClassifier(hidden_layer_sizes=(8, 8, 8),
                     activation='relu',
                     solver='adam',
                     max_iter=500)
mlpc.fit(train, train_target)
predict_train_mlpc = mlpc.predict(train)
predict_test_mlpc = mlpc.predict(test)

net_confusion_matrix = confusion_matrix(train_target, predict_train_mlpc)
print("Confusion matrix train mlpc: \n", net_confusion_matrix)


mlpc_opt = MLPClassifier(hidden_layer_sizes=(20,),
                         activation='tanh',
                         solver='lbfgs',
                         learning_rate='constant',
                         max_iter=10000,
                         alpha=0.001)


mlpc_opt.fit(train, train_target)
predict_train_mlpc_opt = mlpc_opt.predict(train)
predict_test_mlpc_opt = mlpc_opt.predict(test)

net_confusion_matrix_opt = confusion_matrix(test_target, predict_test_mlpc_opt)
class_rep_opt = classification_report(test_target, predict_test_mlpc_opt)
print("Classification report train mlpc_opt: \n", class_rep_opt)
print("Confusion matrix train mlpc_opt: \n", net_confusion_matrix_opt)
# print(tree_confusion_matrix)

cmn = 100*net_confusion_matrix_opt.astype('float') / net_confusion_matrix_opt.sum(axis=1)[:, np.newaxis]  # normalise confusion matrix

f = plt.figure(figsize=(30, 25))
ax = sns.heatmap(cmn,
                 annot=True,
                 fmt=".3f",
                 linewidths=1,
                 vmin=0,
                 vmax=100,
                 square=True,
                 cmap='flare',
                 annot_kws={"size": 100 / np.sqrt(len(net_confusion_matrix_opt))})

plt.ylabel('Actual label %', fontsize=45)
plt.xlabel('Predicted label %', fontsize=45)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=50)
plt.title('MLPClassifier Confusion Matrix', fontsize=50)
f.savefig('/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/NN_conf_matrix.png')

