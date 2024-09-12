import numpy as np
from collections import Counter


def distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._prediction(x) for x in X]
        return predictions

    def _prediction(self, x):
        distances = [distance(x, x_train) for x_train in self.X_train]
        knn_indices = np.argsort(distances)[:self.k]
        knn_labels = [self.y_train[i] for i in knn_indices]

        # majority vote
        most_common = Counter(knn_labels).most_common()
        return most_common[0][0]
