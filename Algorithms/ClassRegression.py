from multiprocessing.sharedctypes import Value
import numpy as np


class ClassRegression:

    def __init__(self, lr=0.001, classification="binary", n_iter=100, epsilon=10e-7):
        self.lr = lr
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
        theta = np.random.rand(m,1)
        if self.classification == "binary":
            pass

        elif self.classification == "softmax":
            pass
    

    def predict(self, X):
        pass


    def _compute_loss_binary(self, y_hat, y):
        return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat), axis=1)

    
    def _compute_loss_softmax(self, y_hat, y):
        """
        Nota bene: y_hat has to be hot-one encoded here (because we do multiclass classification here).
        """
        return -np.mean(np.sum(y*np.log(y_hat, axis=1)))
    
    def _one_hot_encoding(self, y):
        n_classes = np.unique(y).shape[0]
        m = len(y)
        y_one_hot = np.zeros((m, n_classes))
        y_one_hot[np.arrange(m), y] = 1
        return y_one_hot 


    def _gradients_binary(self, theta, X, y):
        activation = self._sigmoid(theta.dot(X))
        return np.mean((activation - y).dot(X))


    def _gradients_softmax(self, theta):
        pass


    def _sigmoid(self, logits):
        return np.exp(1/(1+ np.exp(-logits)))

    def _softmax(self, logits):
        return np.sum(np.exp(logits), axis=1, keepdims=True)













