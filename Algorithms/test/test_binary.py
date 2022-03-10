from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import sys

sys.path.append('../')

from ClassRegression import ClassRegression

def count_accuracy(y_pred, y_test):
    accuracy = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            accuracy += 1
    return accuracy/len(y_test) * 100

X, y = make_classification(n_samples=1500, n_features=2, n_classes=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
X = np.append(np.ones((len(X), 1)), X, axis=1) # adding bias term.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

reg = ClassRegression(classification="binary")
reg.train(X_train, y_train)
y_pred = reg.predict(X_test)

print(count_accuracy(y_pred, y_test)) # Printing the accuracy of a model