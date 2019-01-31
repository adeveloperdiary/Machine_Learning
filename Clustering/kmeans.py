import seaborn as sns
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import copy


class KMeans:
    def __init__(self, k=2):
        self.k = k

    def _get_random_k_points(self, X):
        size = X.shape[1]

        centroids = np.zeros((self.k, size))

        for i in range(size):
            centroids[:, i] = np.random.uniform(np.min(X[:, i]) * 1.1, np.max(X[:, i]) * 0.9, size=self.k)

        return centroids

    def calculate_l2_distance(self, a, b, axis=1):  # axis 1 = by row
        return np.linalg.norm(a - b, axis=axis)

    def fit(self, X):
        self._centroids = self._get_random_k_points(X)
        old_centroids = np.zeros(self._centroids.shape)
        self._clusters = np.zeros(X.shape[0])

        for _ in range(5):
            for i in range(X.shape[0]):
                d = self.calculate_l2_distance(self._centroids, X[i, :])
                self._clusters[i] = np.argmin(d)

            old_centroids = copy.deepcopy(self._centroids)

            for centers in range(self.k):
                points = [X[j, :] for j in range(X.shape[0]) if self._clusters[j] == centers]
                self._centroids[centers,:] = np.mean(points, axis=0)

            error = self.calculate_l2_distance(self._centroids, old_centroids, None)
            print(error)
            if error == 0:
                break



if __name__ == '__main__':
    iris = sns.load_dataset("iris")

    g = sns.pairplot(iris, hue="species", palette="husl")
    plt.show()

    col = ["sepal_length", "sepal_width", "species"]
    X = iris.drop(col, axis=1).values

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
    plt.show()

    model = KMeans(3)
    model.fit(X)

    plt.scatter(X[:, 0], X[:, 1], c=model._clusters, cmap=plt.cm.Set1)
    plt.scatter(model._centroids[:, 0], model._centroids[:, 1], marker='*')
    plt.show()
