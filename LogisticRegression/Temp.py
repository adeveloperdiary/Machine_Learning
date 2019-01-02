import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


class LogisticRegression(object):

    def __init__(self, X, Y):
        if not X.shape[0] == Y.shape[0]:
            raise ValueError('Incompatible training data dimensions.')
        self.X = np.ones((X.shape[0], X.shape[1] + 1))
        self.X[:, :-1] = X
        self.Y = Y.ravel()
        self.theta = None

    def train(self, reg=0.0001, accuracy=0.000001):
        n = self.X.shape[1]
        m = self.X.shape[0]
        theta = np.zeros(n)
        for i in range(1000):
            h = 1./(1 + np.exp(-self.Y*np.sum(np.tile(theta, (m, 1))*self.X, axis=-1)))
            grad = -(1./m) * self.X.T.dot(self.Y*(1 - h)) + reg*theta
            hess = (1./m) * self.X.T.dot(np.dot(np.diag(h*(1-h)), self.X)) + reg*np.diag(np.ones(n))
            delta = -np.linalg.inv(hess).dot(grad)
            theta += delta
        self.theta = theta

    def predict(self, x, probabilities=False):
        if self.theta is None:
            raise ValueError('Model have to be trained before predictions can be made.')
        res = []
        for item in x:
            a = np.array(item)
            x = np.ones(len(item)+1)
            x[:-1] = a
            p = 1./(1 + math.exp(self.theta.dot(x)))
            if p >= 0.5:
                if probabilities:
                    res.append((-1, p))
                else:
                    res.append(-1)
            else:
                if probabilities:
                    res.append((1, p))
                else:
                    res.append(1)
        return res


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

model = LogisticRegression(X_train, y)
model.train()

y_hat = model.predict(X_train)
