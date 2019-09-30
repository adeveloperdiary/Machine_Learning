import numpy as np

import matplotlib.pyplot as plt
from sklearn import preprocessing

import seaborn as sns

iris = sns.load_dataset("iris")

# col = ["sepal_length", "sepal_width"]
col = ["petal_length", "petal_width"]
data = iris.drop(col, axis=1)

le = preprocessing.LabelEncoder()
y = le.fit_transform(data["species"])

y[y == 2] = 1

data = data.drop(["species"], axis=1)

plt.scatter(data["sepal_length"].values, data["sepal_width"].values, c=y, cmap=plt.cm.Set1)
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

plt.scatter(transformed, np.zeros((150)), c=y, cmap=plt.cm.Set1)
plt.show()

M = np.mean(data, axis=0)

colors = ['red', 'blue']
fig, ax = plt.subplots(figsize=(10, 8))
for point, pred in zip(data, y):
    ax.scatter(point[0], point[1], color=colors[pred], alpha=0.15)
    proj = (np.dot(point, W.T) * W) / np.dot(W, W)

    ax.scatter(proj[0], proj[1], color=colors[pred])

plt.show()
