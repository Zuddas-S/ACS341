import numpy as np


class PCAnalysis:

    """
    Computes the simplest form of principal component analysis.
    
    Nota Bene: Standardise your data before introducing it to the algorithm.
    
    """

    def __init__(self, n_components) -> None:
        self.n_components = n_components # principal components are essentially eigenvectors.


    def decompose(self, X : np.ndarray):
        cov_mat = np.cov(X.T)
        eigen_vals, eigen_vectors = np.linalg.eig(cov_mat)
        tot = sum(eigen_vals)
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vectors[:, i]) for i in range(len(eigen_vals))]
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        
        x_dash = self._evaluate_projection(X, eigen_pairs)
        return x_dash
    
    def transform(self, X_test):
        return X_test.dot(self.w)


    def _evaluate_projection(self, X, eigen_pairs):
        self.w = np.hstack((eigen_pairs[j][1][:, np.newaxis] for j in range(self.n_components)))
        x_dash = X.dot(self.w)
        return x_dash

