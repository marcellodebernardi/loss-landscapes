"""
Testing version of the code in examples/loss_contours
"""


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
import loss_landscapes
import loss_landscapes.evaluators.torch as evaluators
from loss_landscapes.utils import deepcopy_model


# constants
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 8


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(IN_DIM, 512)
        self.linear_2 = torch.nn.Linear(512, 256)
        self.linear_3 = torch.nn.Linear(256, 128)
        self.linear_4 = torch.nn.Linear(128, 64)
        self.linear_5 = torch.nn.Linear(64, 32)
        self.linear_6 = torch.nn.Linear(32, 16)
        self.linear_7 = torch.nn.Linear(16, OUT_DIM)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        h = F.relu(self.linear_2(h))
        h = F.relu(self.linear_3(h))
        h = F.relu(self.linear_4(h))
        h = F.relu(self.linear_5(h))
        h = F.relu(self.linear_6(h))
        y = self.linear_7(h)
        return y


class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """
    def __init__(self):
        pass

    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()


def train(model, optimizer, criterion, train_loader, epochs):
    model.train()
    # train model
    for _ in tqdm(range(epochs), 'Training'):
        for count, batch in enumerate(train_loader, 0):
            if count == 100:
                break

            optimizer.zero_grad()
            x, y = batch

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    model.eval()


def main():
    # download MNIST and setup data loaders
    mnist_train = datasets.MNIST(root='../data/', train=True, download=True, transform=Flatten())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)

    # define model and deepcopy initial model
    model = MLP()
    model_initial = deepcopy_model(model, 'torch')
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, optimizer, criterion, train_loader, 2)

    # deepcopy final model and prepare for loss evaluation
    model_final = deepcopy_model(model, 'torch')
    x, y = iter(torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)).__next__()
    evaluator = evaluators.LossEvaluator(criterion, x, y)

    # linear interpolation
    loss_data = loss_landscapes.linear_interpolation(model_initial, model_final, evaluator)
    plt.plot(loss_data)
    plt.title('Linear Interpolation of Loss')
    plt.xlabel('Parameter Step')
    plt.ylabel('Loss')
    plt.show()

    # random direction
    loss_data = loss_landscapes.random_line(model_initial, evaluator, normalization='layer')
    plt.plot(loss_data)
    plt.title('Loss Landscape along Random Direction')
    plt.xlabel('Parameter Step')
    plt.ylabel('Loss')
    plt.show()

    # random plane centered on initialization
    steps = 30
    loss_data = loss_landscapes.random_plane(model_initial, evaluator, steps=steps, normalization='model')
    plt.contour(loss_data)
    plt.title('Loss Contours around Initial Model')
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Loss Contours around Initial Model')
    fig.show()

    # random plane centered on trained model
    loss_data = loss_landscapes.random_plane(model_final, evaluator, steps=30, normalization='model')
    plt.contour(loss_data)
    plt.title('Loss Contours around Trained Model')
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Loss Contours around Trained Model')
    fig.show()


if __name__ == '__main__':
    main()
