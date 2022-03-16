import numpy as np


class PCA:

    """
    Computes the simplest form of principal component analysis.
    
    Nota Bene: Standardise your data before introducing it to the algorithm.
    
    """

    def __init__(self, n_components) -> None:
        self.n_components = n_components # principal components are essentially eigenvectors.


    def decompose(self, X : np.ndarray):
        cov_mat = np.cov(X)
        eigen_vals, eigen_vectors = np.linalg.eig(cov_mat)
        tot = sum(eigen_vals)
        var_exp = [i/tot for i in sorted(eigen_vals, reverse=True)] # relative values of eigenvalues
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vectors[:, i]) for i in range(len(eigen_vals))]
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        
        components = self._evaluate_projection(X, eigen_pairs)
        return components


    def _evaluate_projection(self, X, eigen_pairs):
        w = np.empty([X.shape[0], self.n_components])
        for j in range(self.n_components):
            component = eigen_pairs[j][1][:, np.newaxis]
            np.append(w, component)
        #w = np.hstack(components)
        x_dash = X.T.dot(w)
        return x_dash


# Testing the algorithm
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


Xk, yk = make_classification(n_samples=1500, n_features=10, n_classes=3, n_redundant=0, n_informative=2, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(Xk, yk, test_size=0.3)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

pca = PCA(n_components=2)
print(pca.decompose(X_train_std))

# Need to use some algorithm to evaluate the efficacy of the PCA class.



