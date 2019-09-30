import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns


def load_data():
    iris = sns.load_dataset("iris")
    iris = iris.head(100)
    col = ["sepal_length", "sepal_width"]
    data = iris.drop(col, axis=1)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(data["species"])

    X = data.drop(["species"], axis=1)

    return X.values, y


class LDA:
    def __init__(self):
        pass

    def fit(self, X, y):
        target_classes = np.unique(y)

        mean_vectors = []

        for cls in target_classes:
            mean_vectors.append(np.mean(X[y == cls], axis=0))

        if len(mean_vectors) == 2:
            mu1_mu2 = (mean_vectors[0] - mean_vectors[1]).reshape(1, X.shape[1])
            B = np.dot(mu1_mu2.T, mu1_mu2)

        s_matrix = []

        for cls, mean in enumerate(mean_vectors):
            WB = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == cls]:
                t = (row - mean).reshape(1, X.shape[1])
                WB += np.dot(t.T, t)
            s_matrix.append(WB)

        S = np.zeros((X.shape[1], X.shape[1]))
        for s_i in s_matrix:
            S += s_i

        S_inv = np.linalg.inv(S)

        S_inv_B = S_inv.dot(B)

        eig_vals, eig_vecs = np.linalg.eig(S_inv_B)

        W = eig_vecs[:, 0]

        return W


lda = LDA()
X, y = load_data()
W = lda.fit(X, y)

transformed = X.dot(W)

plt.scatter(transformed, np.zeros((X.shape[0])), c=y, cmap=plt.cm.Set1)
plt.show()

M = np.mean(X, axis=0)

colors = ['red', 'blue']
fig, ax = plt.subplots(figsize=(10, 8))
for point, pred in zip(X, y):
    ax.scatter(point[0], point[1], color=colors[pred], alpha=0.15)
    proj = (np.dot(point, W.T) * W) / np.dot(W, W)

    ax.scatter(proj[0], proj[1], color=colors[pred])

plt.show()

'''
plt.scatter(data["petal_length"].values, data["petal_width"].values, c=y, cmap=plt.cm.Set1)
plt.show()

data = data.values

mu1 = np.mean(data[y == 0], axis=0)
mu2 = np.mean(data[(y == 1) | (y == 2)], axis=0)

mu1_mu2 = (mu1 - mu2).reshape(1, 2)

B = np.dot(mu1_mu2.T, mu1_mu2)

mean_vectors = []
mean_vectors.append(mu1)
mean_vectors.append(mu2)

s_matrix = []

for cls, mean in enumerate(mean_vectors):
    WB = np.zeros((2, 2))
    for row in data[y == cls]:
        t = (row - mean).reshape(1, 2)
        WB += np.dot(t.T, t)
    s_matrix.append(WB)

S = s_matrix[0] + s_matrix[1]

S_inv = np.linalg.inv(S)

S_inv_B = S_inv.dot(B)

eig_vals, eig_vecs = np.linalg.eig(S_inv_B)

W = eig_vecs[:, 0]

transformed = data.dot(W)

plt.scatter(transformed, np.zeros((100)), c=y, cmap=plt.cm.Set1)
plt.show()

M = np.mean(data, axis=0)

colors = ['red', 'blue']
fig, ax = plt.subplots(figsize=(10, 8))
for point, pred in zip(data, y):
    ax.scatter(point[0], point[1], color=colors[pred], alpha=0.15)
    proj = (np.dot(point, W.T) * W) / np.dot(W, W)

    ax.scatter(proj[0], proj[1], color=colors[pred])

plt.show()
'''
