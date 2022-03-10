from multiprocessing.sharedctypes import Value
import numpy as np


class ClassRegression:

    """
    Logistic regression and softmax classification in one class.
    ------------------------------------------------------------------

    Hyperparameters:

    eta - learning rate. This is the hyperparameter that influences the
    rate at which the cost function is updated. It is important to select an appropriate
    learning rate such that it the algorithm doesn't get stuck in local minima.
    Too big, and the learning rate will jump across the solution space
    too abruptly. Too small and the algorithm will take a long time to train.

    classification - used to group instances of data. Can be either
    "binary" or "softmax" in this case.

    n_iter - number of iterations to be executed during the training
    process.

    epsilon - threshold error after which the algorithm will stop
    training.

    ------------------------------------------------------------------

    Methods:

    train(X, y) - training the chosen model. Returns an instance of an
    object, that is the trained model.

    predict(X) - evaluates the labels of an input based on training the
    model had before. Returns an array of labels.

    Other methods are not ment to be called by a user as they are used 
    in the aforementioned functions.

    ------------------------------------------------------------------
    
    """

    def __init__(self, eta=0.007, classification="binary", n_iter=1001, epsilon=10e-4):
        self.eta = eta
        self.classification = classification
        self.n_iter = n_iter
        self.epsilon = epsilon

        classification_types = ["binary", "softmax"]
        if self.classification not in classification_types:
            raise ValueError("This method is not supported. You can use either 'binary' or 'softmax'.")


    def train(self, X: np.ndarray, y: np.ndarray):
        m = len(X)
        n_features = X.shape[1] # number of features in dataset.
        self.theta = np.random.randn(n_features) # initializing weight vector.
        if self.classification == "binary": # checking for type of classification.
            for i in range(self.n_iter):
                y_hat = self._sigmoid(np.dot(X, self.theta))
                loss = self._compute_loss_binary(y_hat, y)
                gradient = np.dot(X.T, (y_hat - y)) / y.size
                self.theta -= self.eta * gradient # updating the weights.
                if i % 10 == 0:
                    print(f"Iteration: {i} Loss: {loss}")
                
                if loss < self.epsilon: # stop training if error is sufficiently low.
                    break

        elif self.classification == "softmax":
            n_outputs = len(np.unique(y))
            self.theta = np.random.randn(n_features, n_outputs)
            y_hot_encoded = self._one_hot_encoding(y)
            for i in range(self.n_iter):
                logits = X.dot(self.theta)
                y_probs = self._softmax(logits)
                loss = self._compute_loss_softmax(y_probs, y_hot_encoded)
                error = y_probs - y_hot_encoded
                gradient = 1.0/m * X.T.dot(error)
                self.theta -= self.eta * gradient

                if i % 10 == 0:
                    print(f"Iteration: {i} Loss: {loss}")
                
                if loss < self.epsilon: # stop training if error is sufficiently low.
                    break

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classification == "binary":
            p_hat = np.where(self._sigmoid(np.dot(X, self.theta)) >= 0.5, 1, 0) # if probability is above 0.5, return 1, else return 0.
        else:
            p_hat = np.argmax(self._softmax(np.dot(X, self.theta)), axis=1)
        return p_hat


    def _compute_loss_binary(self, y_hat: np.ndarray, y: np.ndarray):
        return np.mean((-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))) # binary cross entropy loss function.

    
    def _compute_loss_softmax(self, y_hat: np.ndarray, y: np.ndarray):
        """
        Nota bene: y_hat has to be hot-one encoded here (because we do multiclass classification here).
        """
        return -np.mean(np.sum(y * np.log(y_hat + self.epsilon), axis=1))
    

    def _one_hot_encoding(self, y: np.ndarray):
        n_classes = np.unique(y).shape[0] # number of classes
        m = len(y)
        y_one_hot = np.zeros((m, n_classes))
        y_one_hot[np.arange(m), y] = 1
        return y_one_hot 


    def _sigmoid(self, logits: np.ndarray):
        return 1.0/(1.0 + np.exp(-logits))


    def _softmax(self, logits: np.ndarray):
        return np.exp(logits) / np.sum(np.exp(logits), keepdims=True)



