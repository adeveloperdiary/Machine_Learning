import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

m, n = housing.data.shape

data_scaler = StandardScaler()
data_scaler.fit(housing.data)
data_X = data_scaler.transform(housing.data)

housing_data_plus_bias = np.c_[np.ones((m, 1)), data_X]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")

# Convert vector to column matrix
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

n_epochs = 1000
learning_rate = 0.01

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y

mse = tf.reduce_mean(tf.square(error), name="mse")

# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# gradients = tf.gradients(mse,[theta])[0]

# training_op = tf.assign(theta, theta - learning_rate * gradients)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        sess.run(training_op)

        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE=", mse.eval())

    best_theta = theta.eval()
