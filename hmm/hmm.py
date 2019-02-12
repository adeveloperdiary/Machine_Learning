import pandas as pd
import numpy as np
import copy

data = pd.read_csv('Coding3_HMM_Data.csv')

W = data['Z'].values
X = data['X'].values

# a = np.array(((0.54, 0.46), (0.49, 0.51)))
# b = np.array(((0.16, 0.26, 0.58), (0.25, 0.28, 0.47)))

a = np.array(((0.5, 0.5), (0.5, 0.5)))
b = np.array(((0.1111111, 0.3333333, 0.5555556), (0.1666667, 0.3333333, 0.5000000)))

initial_distribution = np.array((0.5, 0.5))


def forward(X, a, b, initial_distribution):
    alpha = np.zeros((X.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, X[0] - 1]

    for t in range(1, X.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1].T.dot(a[j, :]) * b[j, X[t] - 1]

    return alpha


def backward(X, a, b):
    beta = np.ones((X.shape[0], a.shape[0]))

    for t in range(X.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, X[t + 1] - 1]).T.dot(a[j, :])

    return beta


'''
After 1:

from         A         B
   A 0.4935401 0.5064599
   B 0.4932168 0.5067832

$emissionProbs
      symbols
states         1         2         3
     A 0.1670757 0.2737285 0.5591958
     B 0.2438781 0.2663717 0.4897501


After 100:

from         A         B
   A 0.5381634 0.4618366
   B 0.4866444 0.5133556

$emissionProbs
      symbols
states         1         2         3
     A 0.1627751 0.2625807 0.5746441
     B 0.2514996 0.2778097 0.4706907
'''


def baum_welch_forward_backward(X, a, b, initial_distribution, n_iter=100):
    for _ in range(n_iter):

        alpha = forward(X, a, b, initial_distribution)
        beta = backward(X, a, b)

        gamma = np.zeros((X.shape[0] - 1, a.shape[0], a.shape[0]))

        for t in range(0, X.shape[0] - 1):
            denomenator = ((alpha[t, :].dot(a)) * b[:, X[t + 1] - 1]).dot(beta[t + 1, :].T)
            for h in range(a.shape[0]):
                numerator = alpha[t, h] * a[h, :] * b[:, X[t + 1] - 1] * beta[t + 1, :]
                gamma[t, h, :] = numerator / denomenator

        a_hat = np.sum(gamma, axis=0)
        a_hat = a_hat / np.sum(a_hat, axis=1, keepdims=True)

        new_gamma_for_b = np.zeros((gamma.shape[0] + 1, gamma.shape[1]))

        # Sum the rows
        for t in range(gamma.shape[0]):
            new_gamma_for_b[t,] = np.sum(gamma[t], axis=1)

        # Set the 500th data by copying 498
        new_gamma_for_b[gamma.shape[0]] = np.sum(gamma[gamma.shape[0] - 2], axis=1)

        denom_gamma = np.sum(new_gamma_for_b, axis=0)

        b_hat = copy.deepcopy(b)

        for v in range(1, b.shape[1] + 1):
            b_hat[:, v - 1] = np.sum(new_gamma_for_b[X == v, :], axis=0) / denom_gamma

        a = np.round(copy.deepcopy(a_hat), 7)
        b = np.round(copy.deepcopy(b_hat), 7)

    return np.round(a, 7), np.round(b, 7)


a, b = baum_welch_forward_backward(X, a, b, initial_distribution, n_iter=100)



