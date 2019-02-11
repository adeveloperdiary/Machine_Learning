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


alpha = forward(X, a, b)


def backward(X, a, b):
    beta = np.ones((X.shape[0], a.shape[0]))

    for t in range(X.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, X[t + 1] - 1]).T.dot(a[j, :])

    return beta


beta = backward(X, a, b)

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


