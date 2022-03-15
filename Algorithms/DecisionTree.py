import numpy as np

# Defining a helper class to store data. Classes in Python are data structures.
class Node:
    def __init__(self, feature=None, left=None, right=None, value=None, threshold=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value
        self.threshold = threshold

    
class DecisionTree:
    def __init__(self, min_samples_split=200, max_depth=2, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None


    def train(self, X, y):
        if not self.n_features:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features)

        self.root = self._grow_tree()


    def predict(self, X):
        pass


    def _grow_tree(self, X, y):
        n_samples, n_features = X.shape

        # Now, some stopping criteria have to be implemented

# Defining an entropy to calculate the uncertainty

def entropy(y):
    count = np.bincount(y)
    p = count / len(y)
    return - np.sum([prob * np.log(prob) for prob in p if prob>0])



