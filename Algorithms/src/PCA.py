import numpy as np


class PCA:

    """
    Computes the simplest form of principal component analysis.
    
    Nota Bene: Standardise your data before introducing it to the algorithm.

    ------------------------------------------------------------------
    
    Hyperparameters:

    n_coponents - the number of principal components (i.e eigenvectors)
    is going to be used to map features into a supspace of dimensionality
    N x n_components where N is the number of samples.

    ------------------------------------------------------------------

    Methods:

    fit_decompose(X) - fits the transformation matrix w with respect to
    a introduced data and reduces its dimensionality. Returns the
    dataset of N x n_components dimension.

    decompose(X) - transforms data with already determined transformation
    matrix w. Use only when PCA was fitted before.

    Ex.

    X_std = standardise(X)
    X_train_pca = PCA().fit_decompose(X_train_std)
    X_test_pca = PCA().decompose(X_test_std)

    Other methods are not ment to be called by a user as they are used 
    in the aforementioned functions.

    ------------------------------------------------------------------

    """

    def __init__(self, n_components : int) -> None:
        self.n_components = n_components


    def fit_decompose(self, X : np.ndarray) -> np.ndarray:
        cov_mat = np.cov(X.T) # computing a matrix of covariance.
        eigen_vals, eigen_vectors = np.linalg.eig(cov_mat) # calculating eigenvalues and eigenvectors.
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vectors[:, i]) for i in range(len(eigen_vals))] # pairing eigenvalues correspondigly to their eigenvectors.
        eigen_pairs.sort(key=lambda k: k[0], reverse=True) # sorting in decreasing order so the most significant eigenvectors will be at the beginning.
        x_dash = self._evaluate_projection(X, eigen_pairs)
        return x_dash

    
    def decompose(self, X_test : np.ndarray) -> np.ndarray:
        return X_test.dot(self.w)


    def _evaluate_projection(self, X : np.ndarray, eigen_pairs : list) -> np.ndarray:
        self.w = np.hstack((eigen_pairs[j][1][:, np.newaxis] for j in range(self.n_components))) # stacking the eigenvectors.
        x_dash = X.dot(self.w) # mapping to a new feature space.
        return x_dash

