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
from tests.testing_models.models_torch import MLP


# constants
IN_DIM = 28 * 28
OUT_DIM = 10
LR = 10 ** -1
BATCH_SIZE = 256
EPOCHS = 15
STEPS = 40


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
    x, y = iter(torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=False)).__next__()
    evaluator = evaluators.LossEvaluator(criterion, x, y)

    # # linear interpolation
    # loss_data = loss_landscapes.linear_interpolation(model_initial, model_final, evaluator)
    # plt.plot(loss_data)
    # plt.title('Linear Interpolation of Loss')
    # plt.xlabel('Parameter Step')
    # plt.ylabel('Loss')
    # plt.show()
    #
    # # random direction
    # loss_data = loss_landscapes.random_line(model_initial, evaluator, normalization='layer')
    # plt.plot(loss_data)
    # plt.title('Loss Landscape along Random Direction')
    # plt.xlabel('Parameter Step')
    # plt.ylabel('Loss')
    # plt.show()

    # # random plane centered on initialization
    # loss_data = loss_landscapes.random_plane(model_initial, evaluator, distance=1, steps=steps, normalization='layer')
    # plt.contour(loss_data, levels=50)
    # plt.title('Loss Contours around Initial Model')
    # plt.show()
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # X = np.array([[j for j in range(steps)] for i in range(steps)])
    # Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    # ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.set_title('Loss Contours around Initial Model')
    # fig.show()

    # random plane centered on trained model
    loss_data = loss_landscapes.random_plane(model_final, evaluator, distance=1, steps=STEPS, normalization='layer')
    plt.contour(loss_data, levels=50)
    plt.title('Loss Contours around Trained Model')
    plt.savefig(fname='contour.png', dpi=500)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Loss Contours around Trained Model')
    fig.savefig(fname='contour3d.png', dpi=500)


if __name__ == '__main__':
    main()
