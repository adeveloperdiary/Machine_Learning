import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from A_01_Artificial_Neural_Network.util import get_binary_dataset


class ANN(nn.Module):
    class MyDataLoader(data.Dataset):
        def __init__(self, X, Y):
            self.data = X
            self.target = Y
            self.n_samples = self.data.shape[0]

        def __len__(self):
            return self.n_samples

        def __getitem__(self, index):
            return torch.Tensor(self.data[index]), int(self.target[index])

    def __init__(self, layers_size):
        super(ANN, self).__init__()
        self.layers_size = layers_size
        self.L = len(layers_size)
        self.params = {}

    def initialize_parameters(self):
        for l in range(0, self.L):
            self.add_module("fc" + str(l + 1), nn.Linear(self.layers_size[l], self.layers_size[l + 1]))

    def forward(self, X):

        for l, (name, m) in enumerate(self.named_modules()):
            if l > 0:
                if l == self.L - 1:
                    X = m(X)
                else:
                    X = F.relu(m(X))

        return F.log_softmax(input=X)

    def fit(self, X, Y, learning_rate=0.1, n_iterations=2500):
        self.layers_size.insert(0, X.shape[1])

        self.initialize_parameters()

        train_dataset = self.MyDataLoader(X, Y)
        data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1000, num_workers=1)

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss()
        for epoch in range(n_iterations):
            for k, (data, target) in enumerate(data_loader):
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                forward = self(data)
                loss = criterion(forward, target)
                loss.backward()
                optimizer.step()

            if epoch % 100 == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in data_loader:
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Train Accuracy: {:.2f} %'.format(100 * correct / total))

    def predict(self, X, Y):
        dataset = self.MyDataLoader(X, Y)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1000, num_workers=1)
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in data_loader:
                forward = self(data)
                _, predicted = torch.max(forward.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            print('Accuracy of the network on the {} images: {} %'.format(Y.shape[0], 100 * correct / total))


def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x_orig, train_y_orig, test_x_orig, test_y_orig = get_binary_dataset()
    train_x, train_y, test_x, test_y = pre_process_data(train_x_orig, train_y_orig, test_x_orig, test_y_orig)

    model = ANN(layers_size=[784, 196, 2])
    model.fit(train_x, train_y, learning_rate=0.01, n_iterations=300)
    model.predict(train_x, train_y)
    model.predict(test_x, test_y)
