from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

sys.path.append('../src')

from PCA import PCAnalysis

X, y = make_classification(n_samples=15000, n_features=20, n_classes=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

pca_scratch = PCAnalysis(n_components=6)
X_pca_train_scr = pca_scratch.decompose(X_train_std)
X_pca_test_scr = pca_scratch.transform(X_test_std)


# Need to use some algorithm to evaluate the efficacy of the PCA class.
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
X_pca_train = pca.fit_transform(X_train_std)
X_pca_test = pca.transform(X_test_std)

model = LogisticRegression()
model_pca = LogisticRegression()
model_pca_scratch = LogisticRegression()

# Comparison with model without reduced features, sklearn PCA and from scratch implementation.
model.fit(X_train_std, y_train)
model_pca.fit(X_pca_train, y_train)
model_pca_scratch.fit(X_pca_train_scr, y_train)

print(model.score(X_test_std, y_test))
print(model_pca.score(X_pca_test, y_test))
print(model_pca_scratch.score(X_pca_test_scr, y_test))




