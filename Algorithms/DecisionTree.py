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
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

# Defining an entropy to calculate the uncertainty

def entropy(y):
    count = np.bincount(y)
    p = count / len(y)
    return - np.sum([prob * np.log(prob) for prob in p if prob>0])



