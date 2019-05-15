"""
Testing version of the code in examples/loss_contours
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torchvision.datasets as datasets
from tqdm import tqdm
import loss_landscapes
import loss_landscapes.evaluators.torch as evaluators
from loss_landscapes.utils import deepcopy_model
from tests.testing_utils.models_torch import MLP
from tests.testing_utils.functions_torch import Flatten

# constants
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -2
BATCH_SIZE = 128
EPOCHS = 25
STEPS = 40


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
    # download MNIST and setup data loaders
    mnist_train = datasets.MNIST(root='../data/', train=True, download=True, transform=Flatten())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)
    mnist_test = datasets.MNIST(root='../data', train=False, download=True, transform=Flatten())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10000, shuffle=False)
    x_test, y_test = iter(test_loader).__next__()
    test_evaluator = evaluators.ClassificationAccuracyEvaluator(x_test, y_test)

    # define model and deepcopy initial model
    model = MLP(IN_DIM, OUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    model_initial = deepcopy_model(model, 'torch')
    print('Accuracy: ' + str(loss_landscapes.point(model_initial, test_evaluator)) + '\n')

    train(model, optimizer, criterion, train_loader, EPOCHS)

    # deepcopy final model and prepare for loss evaluation
    model_final = deepcopy_model(model, 'torch')
    print('Accuracy: ' + str(loss_landscapes.point(model_final, test_evaluator)) + '\n')
    x, y = iter(train_loader).__next__()
    loss_evaluator = evaluators.LossEvaluator(criterion, x, y)
    curve_evaluator = evaluators.CurvaturePositivityEvaluator(criterion, x, y)

    # 2D contour plot of loss
    loss_data = loss_landscapes.random_plane(model_final, loss_evaluator, distance=0.5, steps=STEPS, normalization='layer')
    plt.contour(loss_data, levels=50)
    plt.title('Loss Contours around Trained Model')
    plt.savefig(fname='contour.png', dpi=300)

    # 2D color mesh of curvature ratio
    loss_data = loss_landscapes.random_plane(model_final, curve_evaluator, distance=0.5, steps=STEPS, normalization='layer')
    plt.contour(loss_data, levels=50)
    plt.title('Eigenvalue')
    plt.savefig(fname='contour.png', dpi=300)


if __name__ == '__main__':
    main()
