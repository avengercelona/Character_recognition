import extract
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

def display(t):
    a, _ = t
    a = a.numpy()
    plt.imshow(a)

X, Y, X_test, Y_test = extract.get_data()

train_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
test_data = DataLoader(test_dataset, batch_size=64, shuffle=True)

display(next(iter(train_data)))