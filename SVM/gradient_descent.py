import seaborn as sns
from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import copy

iris = sns.load_dataset("iris")
iris = iris.tail(100)

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


N = 200
R_inner = 5
R_outer = 10

# distance from origin is radius + random normal
# angle theta is uniformly distributed between (0, 2pi)
R1 = np.random.randn(N//2) + R_inner
theta = 2*np.pi*np.random.random(N//2)
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(N//2) + R_outer
theta = 2*np.pi*np.random.random(N//2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([ X_inner, X_outer ])
y = np.array([0]*(N//2) + [1]*(N//2))


C = 1.0
lr = 1e-5
n_iters = 600

y[y == 2] = -1

N = X.shape[0]

alphas = np.random.random(N)
b = 0


def linear(X1, X2):
    return np.square(X1).dot(np.square(X2.T))+1
    #return X1.dot(X2.T)


K = linear(X, X)
YY = np.outer(y, y)
YYK = K * YY

losses = []

for _ in range(n_iters):
    loss = np.sum(alphas) - 0.5 * np.sum(YYK * np.outer(alphas, alphas))
    losses.append(loss)

    grad = np.ones(N) - YYK.dot(alphas)
    alphas = alphas + lr * grad

    alphas[alphas < 0] = 0
    alphas[alphas > C] = C

idx = np.where((alphas > 0) & (alphas < C))[0]
bs = y[idx] - (alphas * y).dot(linear(X, X[idx]))

b = np.mean(bs)

plt.plot(losses)
plt.title("loss per iteration")
plt.show()


def _decision_function(x):
    return (alphas * y).dot(linear(X, x)) + b


def predict(X):
    return np.sign(_decision_function(X))


def score(X, Y):
    P = predict(X)
    return np.mean(Y == P)


print(score(X, y))


resolution=100
colors=('b', 'k', 'r')

fig, ax = plt.subplots()

# Generate coordinate grid of shape [resolution x resolution]
# and evaluate the model over the entire space
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
grid = [[_decision_function(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
grid = np.array(grid).reshape(len(x_range), len(y_range))

# Plot decision contours using grid and
# make a scatter plot of training data
ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1, 1, 1),
           linestyles=('--', '-', '--'), colors=colors)
ax.scatter(X[:, 0], X[:, 1],
           c=y, lw=0, alpha=0.3, cmap='seismic')

# Plot support vectors (non-zero alphas)
# as circled points (linewidth > 0)
mask = alphas > 0.
ax.scatter(X[:, 0][mask], X[:, 1][mask],
           c=y[mask], cmap='seismic')

# debug
ax.scatter([0], [0], c='black', marker='x')

plt.show()