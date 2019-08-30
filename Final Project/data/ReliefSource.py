import numpy as np
from sklearn.neighbors import KDTree

class ReliefF(object):

    def __init__(self, n_neighbors=100):
        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.feature_scores = np.zeros(X.shape[1])
        self.tree = KDTree(X)

        for source_index in range(X.shape[0]):
            distances, indices = self.tree.query(
                X[source_index].reshape(1, -1), k=self.n_neighbors+1)

            # Nearest neighbor is self, so ignore first match
            indices = indices[0][1:]

            # Create a binary array that is 1 when the source and neighbor
            #  match and -1 everywhere else, for labels and features..
            labels_match = np.equal(y[source_index], y[indices]) * 2. - 1.
            features_match = np.equal(X[source_index], X[indices]) * 2. - 1.

            # The change in feature_scores is the dot product of these  arrays
            self.feature_scores += np.dot(features_match.T, labels_match)

        self.top_features = np.argsort(self.feature_scores)[::-1]


    def transform(self, X ,n_features_to_keep):
        return X[:, self.top_features[:n_features_to_keep]]
