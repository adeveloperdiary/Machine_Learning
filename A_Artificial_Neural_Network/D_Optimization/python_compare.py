import A_Artificial_Neural_Network.B_BatchGradientDescent.python_mini_batch_gd as bgd
import A_Artificial_Neural_Network.D_Optimization.python_adam as adam
import A_Artificial_Neural_Network.D_Optimization.python_momentum as momentum
import A_Artificial_Neural_Network.D_Optimization.python_rmsprop as rmsprop
import matplotlib.pylab as plt
import datasets.mnist.loader as mnist
import numpy as np
from A_Artificial_Neural_Network.util import pre_process_data


train_x, train_y, test_x, test_y = mnist.get_data(True)

train_x, train_y, test_x, test_y = pre_process_data(train_x, train_y, test_x, test_y)


print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

costs = []
layers_dims = [64, 10]

batch_gd = bgd.ANN(layers_dims)
batch_gd.fit(train_x, train_y, learning_rate=0.001, n_iterations=1000)

costs.append(batch_gd.costs)

momentum_gd = momentum.ANN(layers_dims, batch_size=64, learning_rate=0.001)
momentum_gd.fit(train_x, train_y, n_iterations=1000)

costs.append(momentum_gd.costs)

rmsprop_gd = rmsprop.ANN(layers_dims, batch_size=64, learning_rate=0.001)
rmsprop_gd.fit(train_x, train_y, n_iterations=1000)

costs.append(rmsprop_gd.costs)

adam_gd = adam.ANN(layers_dims, batch_size=64, learning_rate=0.001)
adam_gd.fit(train_x, train_y, n_iterations=1000)

costs.append(adam_gd.costs)

plt.figure()

for cost in costs:
    plt.plot(np.arange(len(cost)), cost)

plt.legend(['batch gd', 'momentum', 'rmsprop', 'adam'], loc='upper right')
plt.xlabel("epochs")
plt.ylabel("cost")
plt.show()
