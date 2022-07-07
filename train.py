import extract
import numpy as np
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


#prepare tensors
X, Y, X_test, Y_test = extract.get_data()

X = torch.Tensor(np.array(X))
X_test = torch.Tensor(np.array(X_test))

m = X.shape[0]

index = torch.tensor(Y, dtype=torch.int64)
index = torch.unsqueeze(index, 1)
Y = torch.zeros(m, 26, dtype=index.dtype).scatter(1, index, value=1)

index2 = torch.tensor(Y_test, dtype=torch.int64)
index2 = torch.unsqueeze(index2, 1)
Y_test = torch.zeros(X_test.shape[0], 26, dtype=index2.dtype).scatter(1, index2, value=1)

#prepare dataset

train_dataset = TensorDataset(X, Y)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test, Y_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

display(train_dataset[0])