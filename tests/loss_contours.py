import os
import sys
import copy
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.evaluators
import loss_landscapes.evaluators.torch


# input dimension and output dimension for an MNIST classifier
IN_DIM = 28 * 28
OUT_DIM = 10
# training settings
LR = 10 ** -3
EPOCHS = 1
BATCH_SIZE = 64


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(IN_DIM, 512)
        self.linear_2 = torch.nn.Linear(512, 256)
        self.linear_3 = torch.nn.Linear(256, 128)
        self.linear_4 = torch.nn.Linear(128, 64)
        self.linear_5 = torch.nn.Linear(64, OUT_DIM)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = F.relu(self.linear_4(x))
        x = self.softmax(self.linear_5(x))
        return x


class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """

    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()


def train(model, optimizer, criterion, train_loader, batches=100, epochs=1):
    # save initial state
    model_initial = copy.deepcopy(model)

    # train model
    for epoch in range(epochs):
        for count, batch in enumerate(train_loader, 0):
            if count == batches:
                break

            x, y = batch
            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    # save final state
    model_final = copy.deepcopy(model)

    return model_initial, model_final


def main():
    # download MNIST
    mnist_train = datasets.MNIST(root='../data/', train=True, download=True, transform=Flatten())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)

    # define model
    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    # train model
    model_initial, model_final = train(model, optimizer, criterion, train_loader, 100, 1)

    # collect linear interpolation data
    x, y = iter(torch.utils.data.DataLoader(mnist_train, batch_size=10000, shuffle=False)).__next__()
    evaluator = LossEvaluator(criterion, x, y)
    loss_data = loss_landscapes.linear_interpolation(model_initial, model_final, evaluator)

    # plot linear interpolation
    plt.plot(loss_data)
    plt.title('Linear Interpolation of Loss')
    plt.xlabel('Parameter Step')
    plt.ylabel('Loss')
    plt.show()

    # collect planar data
    loss_data = loss_landscapes.random_plane(model_initial, evaluator, steps=20)

    # plot planar data
    plt.contourf(loss_data)
    plt.title('Contour Plot of Loss Landscape')
    plt.show()


if __name__ == '__main__':
    main()
