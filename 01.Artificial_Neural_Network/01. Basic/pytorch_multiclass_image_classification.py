import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import get_binary_dataset, pre_process_data
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 196)
        self.fc3 = nn.Linear(196, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(input=x,dim=0)


class MyDataLoader(data.Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.target = Y
        self.n_samples = self.data.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        print(index,self.target[index])
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])


if __name__ == '__main__':

    train_x_orig, train_y_orig, test_x_orig, test_y_orig = get_binary_dataset()
    train_x, train_y, test_x, test_y = pre_process_data(train_x_orig, train_y_orig, test_x_orig, test_y_orig)

    my_train_loader = MyDataLoader(train_x, train_y)

    my_loader = train_data = data.DataLoader(my_train_loader,batch_size=1)

    my_model = ANN()
    optimizer = optim.SGD(my_model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    for epoch in range(100):
        for k, (data, target) in enumerate(my_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            pred=my_model(data)
            print(pred)
            print(target)
            loss = criterion(pred, target.view(-1).long())
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.data[0]))
