from multiprocessing.sharedctypes import Value
import numpy as np


class ClassRegression:

    def __init__(self, eta=0.0001, classification="binary", n_iter=1000, epsilon=10e-8):
        self.eta = eta
        self.classification = classification
        self.n_iter = n_iter
        self.epsilon = epsilon

        classification_types = ["binary", "softmax"]
        if self.classification not in classification_types:
            raise ValueError("This method is not supported. You can use either 'binary' or 'softmax'.")

    def train(self, X, y):
        # Initialize the weights.
        # Train over the training data set
        # Iterate
        # Return weight after finishing
        m = len(X)
        n_outputs = X.shape[1]
        self.theta = np.random.rand(n_outputs, 1)
        if self.classification == "binary":
            for i in range(self.n_iter):
                y_hat = self._sigmoid(np.dot(X, self.theta))
                loss = self._compute_loss_binary(y_hat, y)
                self.theta = self.theta - self.eta * np.mean((y_hat - y).dot(X))
                if i % 10 == 0:
                    print(f"Iteration: {i} Loss: {loss}")
                
                if loss < self.epsilon:
                    break

        elif self.classification == "softmax":
            pass
        
        return self

    def predict(self, X):
        p = np.around(self._sigmoid(np.dot(X, self.theta)))
        return p


    def _compute_loss_binary(self, y_hat, y):
        m = len(y)
        cost = (1/m)*(((-y).T @ np.log(y_hat + self.epsilon))-((1-y).T @ np.log(1-y_hat + self.epsilon)))
        return cost

    
    def _compute_loss_softmax(self, y_hat, y):
        """
        Nota bene: y_hat has to be hot-one encoded here (because we do multiclass classification here).
        """
        return -np.mean(np.sum(y*np.log(y_hat, axis=1)))
    
    def _one_hot_encoding(self, y):
        n_classes = np.unique(y).shape[0]
        m = len(y)
        y_one_hot = np.zeros((m, n_classes))
        y_one_hot[np.arange(m), y] = 1
        return y_one_hot 


    def _gradients_binary(self, theta, X, y):
        activation = self._sigmoid(theta.dot(X))
        return np.mean((activation - y).dot(X))


    def _gradients_softmax(self, theta):
        pass


    def _sigmoid(self, logits):
        return 1/(1 + np.exp(-logits))

    def _softmax(self, logits):
        return np.sum(np.exp(logits), axis=1, keepdims=True)

## Training and testing the algorithm

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1500, n_features=2, n_classes=2, n_redundant=0)
X = np.append(np.ones((len(X), 1)), X, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
reg = ClassRegression()
reg.train(X_train, y_train)
y_pred = reg.predict(X_test)

def count_accuracy(y_pred, y_test):
    accuracy = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            accuracy += 1
    return accuracy/len(y_test) * 100

print(count_accuracy(y_pred, y_test))







