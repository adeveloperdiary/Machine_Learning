# -*- coding: utf-8 -*-

"""
Author: Ashish Verma
This code does logistic regression using newton's method for binary response variable
This code was developed to give a clear understanding of what goes behind the curtains in logistic regression using newton's method.
Feel free to use/modify/improve/etc. but beware that this may not be efficient for production related usage (especially where data is large).

This code is developed only for a binary response variable. If the classes are other than 0 and 1, then you'll have to transform them.
"""

import pandas as pd
import numpy as np
import numpy.linalg as linalg
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


class logisticRegression:
    def __init__(self, X):
        # Estimated parameters are stored in 'newEstimatedParameters'. An extra element has been increased to store the intercept
        self.newEstimatedParameters = np.zeros(X.shape[1] + 1)
        self.oldEstimatedParameters = []
        # self.probability is where final probability of the data is stored after convergence
        self.probability = []

    # This function contains one iteration update for parameters using newton's method.
    def newtonMethodForEstimation(self, firstDerivative, secondDerivative):
        # Implementation of β_(t+1) = β_t - (f'(β_t))/(f''(β_t)), where t is current iteration
        # Parameters are updated using the new delta
        self.newEstimatedParameters = self.newEstimatedParameters - (np.dot(linalg.inv(secondDerivative), firstDerivative))

    # Calculate first derivative of log likelihood which is ∂l/(∂β_j ) = Σ_(i=1)^N (y_i- P(x_i ))x_ij
    def calculateFirstDerivative(self, X, Y, probability):
        firstDerivative = np.dot((Y - probability), X)
        return firstDerivative

    # Calculate second derivative of the log liklihood function which is (∂^2 l)/(∂β_j ∂β_k ) = -Σ_(i=1)^N (x_ij)(x_ik) P(x_i )(1- P(x_i ))
    def calculateSecondDerivative(self, X, probability):
        probMul = probability * (1 - probability)
        xProb = np.array([x * y for (x, y) in zip(X, probMul)])
        secondDerivative = -1 * np.dot(xProb.T, X)
        return secondDerivative

    # Calculate probability which was of this form for a single feature - p(x)=e^(β_0+ β_1 x)/(1+e^(β_0+ β_1 x) )
    # Below function calculates it for multiple features
    def calculateProbability(self, X):
        probability = 1 / (1 + np.exp(-np.dot(self.newEstimatedParameters, X.T)))
        return probability

    # This is the main function which fits the data and estimates the parameters of logistic regression
    def logisticRegressionFit(self, X, Y, maxIteration, diffThreshold):
        # Add a dummy feature with value 1 for calculating intercept
        dummyParameter = np.ones((X.shape[0], 1))
        X = np.c_[X, dummyParameter]
        # Initialize the iteration number
        iteration = 0
        # List which contains the incremental update to parameters over iterations
        # Will help us in looking at the convergence later
        diffParametersList = []

        # This is the main convergence loop which checks if the parameter values conerged
        while (list(self.newEstimatedParameters) != list(self.oldEstimatedParameters)):
            # Store old parameter values
            self.oldEstimatedParameters = self.newEstimatedParameters
            # Calculate probability
            probability = self.calculateProbability(X)
            # Calculate first derivative
            firstDerivative = self.calculateFirstDerivative(X, Y, probability)
            # Calculate second derivative
            secondDerivative = self.calculateSecondDerivative(X, probability)
            # Update parameter values using newton's method
            self.newtonMethodForEstimation(firstDerivative, secondDerivative)
            # Increment iteration value
            iteration = iteration + 1
            # Calculate increment in parameter values over one iteration
            diffValue = linalg.norm(self.newEstimatedParameters - self.oldEstimatedParameters)
            diffParametersList.append(diffValue)
            # Check if the maximum number of iterations have completed. If yes then break.
            if (iteration > maxIteration):
                print
                "maximum number of iterations reached. Breaking !"
                break
            # Check if the threshold for iterative update has reached. If yes then break.
            else:
                if (diffValue <= diffThreshold):
                    print
                    "diffThreshold value breached so breaking !"
                    break
        # Update final probability value
        self.probability = probability
        return diffParametersList

    # Predict the binary outcome by calculating the probability of the test data using parameters estimated
    def predictClasses(self, X):
        # Add dummy variable with value 1 to be multiplied with intercept term
        dummyParameter = np.ones((X.shape[0], 1))
        X = np.c_[X, dummyParameter]
        predictedProb = self.calculateProbability(X)
        predictedClasses = [0 if x <= 0.5 else 1 for x in predictedProb]
        return predictedClasses


iris = sns.load_dataset("iris")
iris = iris.tail(100)

# g = sns.pairplot(iris, hue="species", palette="husl")
# plt.show()

# col = ["sepal_length", "sepal_width"]
# data = iris.drop(col, axis=1)

# data=iris

le = preprocessing.LabelEncoder()
y = le.fit_transform(iris["species"])

plt.scatter(iris["sepal_length"].values, iris["petal_width"].values, c=y, cmap=plt.cm.Set1)
plt.show()

X_train = iris.drop(["species"], axis=1).values

model = logisticRegression(X_train)
model.logisticRegressionFit(X_train, y, 5, 0.5)

y_hat = model.predictClasses(X_train)

print((y_hat == y).mean())

