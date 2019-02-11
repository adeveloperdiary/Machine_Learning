import tensorflow as tf
import numpy as np

# Zero Tensor
zero_tsr = tf.zeros([2, 2])

with tf.Session() as sess:
    print(sess.run(zero_tsr))

# One Tensor
ones_tsr = tf.ones([2, 2])

with tf.Session() as sess:
    print(sess.run(ones_tsr))

# Any fill Tensor
filled_tsr = tf.fill([2, 2], 4)

with tf.Session() as sess:
    print(sess.run(filled_tsr))

# Array to tensor

const_tsr = tf.constant([1, 2, 3])

with tf.Session() as sess:
    print(sess.run(const_tsr))

# tensor with defined intervals ( does not include the limit value )

linear_tsr = tf.range(start=0, limit=10, delta=2)

with tf.Session() as sess:
    print(sess.run(linear_tsr))

# Random tensor from normal dist
rand_unif_tsr = tf.random_normal([10, 10], mean=0.0, stddev=1.0)

with tf.Session() as sess:
    print(sess.run(rand_unif_tsr))

shuffled_output = tf.random_shuffle(rand_unif_tsr)

# tf.random_crop()

my_var = tf.Variable(tf.zeros([2, 2]))

# for converting numpy array to tensor
# tf.convert_to_tensor()

my_var = tf.Variable(tf.zeros([2, 2]))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=[2, 2])
    y = tf.identity(x)
    x_vals = np.random.rand(2, 2)
    sess.run(y, feed_dict={x: x_vals})

sess = tf.Session()
identity_matrix = tf.diag([1.0, 1.0, 1.0])
A = tf.truncated_normal([2, 3])
B = tf.fill([2, 3], 5.0)
C = tf.random_uniform([3, 2])
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(identity_matrix))
print(sess.run(A))
print(sess.run(B))
print(sess.run(C))
print(sess.run(D))
print(sess.run(A + B))
print(sess.run(B - B))
print(sess.run(tf.matmul(B, identity_matrix)))
print(sess.run(tf.transpose(C)))
print(sess.run(tf.matrix_determinant(D)))
print(sess.run(tf.matrix_inverse(D)))
print(sess.run(tf.self_adjoint_eig(D)))
