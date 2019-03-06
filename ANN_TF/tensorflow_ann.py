import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten / 255.
X_test = X_test_flatten / 255.

Y_train = np.eye(6)[Y_train_orig.reshape(-1)].T
Y_test = np.eye(6)[Y_test_orig.reshape(-1)].T

ops.reset_default_graph()
tf.set_random_seed(1)
seed = 3

(n_x, m) = X_train.shape
n_y = Y_train.shape[0]

costs = []

# Placeholder for input data
X = tf.placeholder(tf.float32, shape=[X_train.shape[0], None], name='X')
Y = tf.placeholder(tf.float32, shape=[Y_train.shape[0], None], name='Y')

tf.set_random_seed(1)

# initialize variables
W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

# Forward Propagation
Z1 = tf.add(tf.matmul(W1, X), b1)
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(W2, A1), b2)
A2 = tf.nn.relu(Z2)
Z3 = tf.add(tf.matmul(W3, A2), b3)

# Cost Calculation
logits = tf.transpose(Z3)
labels = tf.transpose(Y)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

# Back-Propagation
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(1500):
        epoch_cost = 0.

        num_minibatches = int(m / 32)
        seed = seed + 1
        minibatches = tf_utils.random_mini_batches(X_train, Y_train, mini_batch_size=32, seed=seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

            epoch_cost += minibatch_cost / num_minibatches

        if epoch % 100 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if epoch % 5 == 0:
            costs.append(epoch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(0.0001))
    plt.show()

    sess.run([W1, b1, W2, b2, W3, b3])
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
