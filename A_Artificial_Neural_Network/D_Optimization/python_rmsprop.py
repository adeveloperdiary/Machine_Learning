import numpy as np
import datasets.mnist.loader as mnist
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
import math
from A_Artificial_Neural_Network.util import get_binary_dataset


class ANN:
    def __init__(self, layers_size, batch_size=64, learning_rate=0.1):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.costs = []
        self.mini_batch_size = batch_size
        self.velocity = {}
        self.learning_rate = learning_rate
        self.beta = 0.999

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def initialize_parameters(self):
        np.random.seed(1)

        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / np.sqrt(
                self.layers_size[l - 1])
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))

            self.velocity["dW" + str(l)] = np.zeros(
                [self.parameters["W" + str(l)].shape[0], self.parameters["W" + str(l)].shape[1]])
            self.velocity["db" + str(l)] = np.zeros(
                [self.parameters["b" + str(l)].shape[0], self.parameters["b" + str(l)].shape[1]])

    def forward(self, X):
        store = {}

        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = self.relu(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z

        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.softmax(Z)
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z

        return A, store

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def relu_derivative(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ

    def backward(self, X, Y, store):

        derivatives = {}
        n = X.shape[0]

        store["A0"] = X.T

        A = store["A" + str(self.L)]
        dZ = A - Y.T

        dW = dZ.dot(store["A" + str(self.L - 1)].T) / n
        db = np.sum(dZ, axis=1, keepdims=True) / n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)

        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db

        for l in range(self.L - 1, 0, -1):
            dZ = self.relu_derivative(dAPrev, store["Z" + str(l)])
            dW = 1. / n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)

            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db

        return derivatives

    def get_mini_batches(self, X, Y, seed):
        m = X.shape[0]  # number of training examples
        mini_batches = []
        np.random.seed(seed)

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(
            m / self.mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * self.mini_batch_size: k * self.mini_batch_size + self.mini_batch_size, :]
            mini_batch_Y = shuffled_Y[k * self.mini_batch_size: k * self.mini_batch_size + self.mini_batch_size, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % self.mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * self.mini_batch_size: m, :]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * self.mini_batch_size: m, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def minibatch_gd_using_rmsprop(self, derivatives):
        for l in range(1, self.L + 1):
            self.velocity["dW" + str(l)] = self.beta * self.velocity["dW" + str(l)] + (1 - self.beta) * np.square(
                derivatives["dW" + str(l)])
            self.velocity["db" + str(l)] = self.beta * self.velocity["db" + str(l)] + (1 - self.beta) * np.square(
                derivatives["db" + str(l)])

            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - (
                    self.learning_rate / (np.sqrt(self.velocity["dW" + str(l)]) + 1e-8)) * derivatives[
                                                "dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - (
                    self.learning_rate / (np.sqrt(self.velocity["db" + str(l)]) + 1e-8)) * derivatives[
                                                "db" + str(l)]

    def fit(self, X, Y, n_iterations=2500):
        np.random.seed(1)

        self.layers_size.insert(0, X.shape[1])

        self.initialize_parameters()
        for epoch in range(n_iterations):

            cost = 0.

            minibatches = self.get_mini_batches(X, Y, seed=epoch)

            num_batches = len(minibatches)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                A, store = self.forward(minibatch_X)
                cost += -np.mean(minibatch_Y * np.log(A.T + 1e-8)) / num_batches
                derivatives = self.backward(minibatch_X, minibatch_Y, store)

                self.minibatch_gd_using_rmsprop(derivatives)

            if epoch % 100 == 0:
                print("Cost: ", cost, "Train Accuracy:", self.predict(X, Y))

            if epoch % 10 == 0:
                self.costs.append(cost)

    def predict(self, X, Y):
        A, cache = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()


def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))

    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = mnist.get_data()
    # train_x, train_y, test_x, test_y = get_binary_dataset()

    train_x, train_y, test_x, test_y = pre_process_data(train_x, train_y, test_x, test_y)

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    layers_dims = [50, 10]

    ann = ANN(layers_dims, batch_size=64, learning_rate=0.001)
    ann.fit(train_x, train_y, n_iterations=500)
    print("Train Accuracy:", ann.predict(train_x, train_y))
    print("Test Accuracy:", ann.predict(test_x, test_y))
    ann.plot_cost()
