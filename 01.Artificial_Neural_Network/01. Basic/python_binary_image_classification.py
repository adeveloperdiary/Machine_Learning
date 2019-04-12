import numpy as np
import h5py
import datasets.mnist.loader as mnist


class ANN:
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size) - 1
        self.n = 0

    def relu(self, Z):
        A = np.maximum(0, Z)
        return A

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def initialize_parameters(self):
        np.random.seed(1)

        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))

    def forward(self, X):
        cache = {}

        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.sigmoid(Z)
            cache["A" + str(l + 1)] = A
            cache["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            cache["Z" + str(l + 1)] = Z

        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.sigmoid(Z)
        cache["A" + str(self.L)] = A
        cache["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        cache["Z" + str(self.L)] = Z

        return A, cache

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def relu_derivative(self, dA, Z):

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ

    def backward(self, X, Y, cache):

        derivatives = {}

        cache["A0"] = X.T

        A = cache["A" + str(self.L)]
        dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)

        dZ = dA * self.sigmoid_derivative(cache["Z" + str(self.L)])
        dW = dZ.dot(cache["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = cache["W" + str(self.L)].T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(cache["Z" + str(l)])
            dW = 1. / self.n * dZ.dot(cache["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = cache["W" + str(l)].T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives

    def fit(self, X, Y, learning_rate=0.01, n_iterations=2500):
        np.random.seed(1)

        self.n = X.shape[0]
        self.initialize_parameters()
        for loop in range(n_iterations):
            A, cache = self.forward(X)
            cost = np.squeeze(-(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))) / self.n)
            derivatives = self.backward(X, Y, cache)

            for l in range(1, self.L + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]

            if loop % 100 == 0:
                print(cost)

    def predict(self, X, Y):
        A, cache = self.forward(X)
        n = X.shape[0]
        p = np.zeros((1, n))

        for i in range(0, A.shape[1]):
            if A[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        print("Accuracy: " + str(np.sum((p == Y) / n)))


def get_binary_dataset(train_x_orig, train_y_orig, test_x_orig, test_y_orig):
    index_5 = np.where(train_y_orig == 5)
    index_8 = np.where(train_y_orig == 8)

    index = np.concatenate([index_5[0], index_8[0]])
    np.random.seed(1)
    np.random.shuffle(index)

    train_y = train_y_orig[index]
    train_x = train_x_orig[index]

    train_y[np.where(train_y == 5)] = 0
    train_y[np.where(train_y == 8)] = 1

    index_5 = np.where(test_y_orig == 5)
    index_8 = np.where(test_y_orig == 8)

    index = np.concatenate([index_5[0], index_8[0]])
    np.random.shuffle(index)

    test_y = test_y_orig[index]
    test_x = test_x_orig[index]

    test_y[np.where(test_y == 5)] = 0
    test_y[np.where(test_y == 8)] = 1

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x_orig, train_y_orig, test_x_orig, test_y_orig = mnist.get_data()
    train_x, train_y, test_x, test_y = get_binary_dataset(train_x_orig, train_y_orig, test_x_orig, test_y_orig)

    m_train = train_x.shape[0]
    num_px = train_x.shape[1]
    m_test = test_x.shape[0]

    train_x_flatten = train_x.reshape(train_x.shape[0], -1)
    test_x_flatten = test_x.reshape(test_x.shape[0], -1)

    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    # layers_dims = [12288, 20, 7, 5, 1]
    layers_dims = [784, 196, 1]

    ann = ANN(layers_dims)
    ann.fit(train_x, train_y, learning_rate=0.1, n_iterations=2000)
    ann.predict(train_x, train_y)
    ann.predict(test_x, test_y)
