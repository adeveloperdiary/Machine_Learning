import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


class LogisticRegressionUsingGradientDescent:
    def __init__(self, learning_rate=0.01, iteration=1000, intercept=True):
        self.lr = learning_rate
        self.iteration = iteration
        self.intercept = intercept

        self.w = []
        self.n = 0

        self.X = []
        self.y = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost(self, h, y):
        return (1 / self.n) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))

    def _gradient(self, h, y):
        return (1 / self.n) * (np.dot(self.X.T, (h - y)))

    def _add_intercept_to_x(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y):
        self.n = X.shape[0]

        self.X = X
        self.y = y

        if self.intercept:
            self.X = self._add_intercept_to_x(self.X)

        self.w = np.zeros((self.X.shape[1]))

        for i in range(self.iteration):
            z = np.dot(self.X, self.w)
            h = self._sigmoid(z)

            gradient = self._gradient(h, self.y)
            self.w = self.w - self.lr * gradient

            if i % 100 == 0:
                z = np.dot(self.X, self.w)
                h = self._sigmoid(z)
                print("Loss: {:f} , Accuracy: {:f}".format(self._cost(h, self.y), self._accuracy()))

    def predict(self, X, probability=False, threshold=0.5):
        if self.intercept:
            X = self._add_intercept_to_x(X)

        z = np.dot(X, self.w)
        if probability:
            return self._sigmoid(z)
        else:
            return np.where(self._sigmoid(z) >= threshold, 1, 0)

    def accuracy(self, X, y, threshold=0.5):
        y_hat = self.predict(X, threshold=threshold)
        return (y_hat == y).mean()

    def _accuracy(self, threshold=0.5):
        z = np.dot(self.X, self.w)
        y_hat = np.where(self._sigmoid(z) >= threshold, 1, 0)
        return (y_hat == self.y).mean()


if __name__ == '__main__':
    iris = sns.load_dataset("iris")
    iris = iris.head(100)

    # g = sns.pairplot(iris, hue="species", palette="husl")
    # plt.show()

    col = ["sepal_length", "sepal_width"]
    data = iris.drop(col, axis=1)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(data["species"])

    # plt.scatter(data["petal_length"].values, data["petal_width"].values, c=y, cmap=plt.cm.Set1)
    # plt.show()

    X_train = data.drop(["species"], axis=1).values

    model = LogisticRegressionUsingGradientDescent()
    model.fit(X_train, y)

    y_hat = model.predict(X_train)
    y_hat = model.predict(X_train, probability=True)

    model.accuracy(X_train, y)
