"""
Testing version of the code in examples/loss_contours
"""


import matplotlib.pyplot as plt
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
BATCH_SIZE = 64


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(IN_DIM, 512)
        self.linear_2 = torch.nn.Linear(512, 256)
        self.linear_3 = torch.nn.Linear(256, OUT_DIM)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        h = F.relu(self.linear_2(h))
        y = self.linear_3(h)
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
            optimizer.zero_grad()
            x, y = batch

            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

    model.eval()


def main():
    # download MNIST
    mnist_train = datasets.MNIST(root='../data/', train=True, download=True, transform=Flatten())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)
    mnist_test = datasets.MNIST(root='../data/', train=True, download=False, transform=Flatten())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10000, shuffle=False)

    # define model and deepcopy initial model
    model = MLP()
    model_initial = deepcopy_model(model, 'torch')

    x, y = iter(test_loader).__next__()
    evaluator = evaluators.ClassificationAccuracyEvaluator(x, y)
    print('Classification accuracy before: ' + str(loss_landscapes.point(model, evaluator)))

    # train model
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, optimizer, criterion, train_loader, 25)

    print('Classification accuracy after: ' + str(loss_landscapes.point(model, evaluator)))

    # deepcopy final model
    model_final = deepcopy_model(model, 'torch')

    # collect linear interpolation data
    x, y = iter(torch.utils.data.DataLoader(mnist_train, batch_size=10000, shuffle=False)).__next__()
    evaluator = evaluators.LossEvaluator(criterion, x, y)
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
