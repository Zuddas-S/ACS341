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

X, y = make_classification(n_samples=1500, n_features=2, n_classes=3, n_redundant=0, n_informative=2, n_clusters_per_class=1)
X = np.append(np.ones((len(X), 1)), X, axis=1) # adding bias term.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # Splitting data with 70/30 proportion.

reg_soft = ClassRegression(classification="softmax")
reg_soft.train(X_train, y_train)
y_hat = reg_soft.predict(X_test)
print(count_accuracy(y_hat, y_test))


