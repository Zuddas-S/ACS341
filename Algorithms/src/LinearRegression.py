import numpy as np


class LinearRegression:

    """
    A simple linear regression algorithm.

    ------------------------------------------------------------------

    Hyperparameters:

    n_iter - number of iterations to be executed during the training
    process.

    eta - learning rate. This is the hyperparameter that influences the
    rate at which the cost function is updated. It is important to select 
    an appropriate learning rate such that it the algorithm doesn't get stuck 
    in local minima. Too big, and the learning rate will jump across the 
    solution space too abruptly. Too small and the algorithm will take a long 
    time to train.
    
    ------------------------------------------------------------------

    Methods:

    train(X, y) - training the linear regression model. Returns an 
    instance of an object, that is the trained model. 

    predict(X) - evaluates the labels of an input based on training the
    model had before. Returns an array of labels.

    Other methods are not ment to be called by a user as they are used 
    in the aforementioned functions.

    ------------------------------------------------------------------
    
    """

    def __init__(self, n_iter=1000, eta=0.01):
        self.theta = 0
        self.n_iter = n_iter
        self.eta = eta


    def train(self, X, y):
        self._initialize_weights(X)
        self.error_iter = []
        self.theta_iter = []
        for i in range(self.n_iter):
            loss = self._compute_loss(X, y)
            self.theta = self.theta - self.eta*loss # updating the weights by subtracting it by a gradient of the loss function.
            y_hat = np.dot(X, self.theta) # not an actual prediction but necessary for calculation of MSE.
            error = self._compute_mse(y_hat, y)
            self.error_iter.append(error)
            self.theta_iter.append(self.theta)
        return self
    

    def predict(self, X):
        return np.dot(X, self.theta)


    def _initialize_weights(self, X):
        n_outputs = X.shape[1]
        self.theta =  np.random.rand(n_outputs, 1)*10


    def _compute_loss(self, X, y):
        m = len(X)
        return 2.0/m * np.dot(X.T, (np.dot(X, self.theta) - y))


    def _compute_mse(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)


