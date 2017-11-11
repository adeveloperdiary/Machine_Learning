import numpy as np
import matplotlib.pyplot as plt

N = 50

# Get data point from Uniform Distribution
X = np.linspace(0, 10, N)

# Y will be Normal Distribution
y = 0.5 * X + np.random.randn(N)

# Set outliers
y[-1] += 30
y[-2] += 30

plt.scatter(X, y, alpha=.5)
plt.show()

X = np.vstack([np.ones(N), X]).T

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(y))

Yhat_ml = X.dot(w_ml)

plt.scatter(X[:, 1], y, alpha=0.5)
plt.plot(X[:, 1], Yhat_ml)
plt.show()

# L2

l2 = 1000.0
w_map = np.linalg.solve((l2 * np.eye(2) + X.T.dot(X)), X.T.dot(y))

Yhat_map = X.dot(w_map)

plt.scatter(X[:, 1], y, alpha=0.5)
plt.plot(X[:, 1], Yhat_ml, label="ML", color="red")
plt.plot(X[:, 1], Yhat_map, label="MAP", color="green")
plt.legend()
plt.show()
