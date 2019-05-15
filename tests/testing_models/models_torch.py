import torch
import torch.nn
import torch.nn.functional as F
import torchvision.models as models


class MLP(torch.nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 512)
        self.linear_2 = torch.nn.Linear(512, 256)
        self.linear_3 = torch.nn.Linear(256, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        h = F.relu(self.linear_2(h))
        y = self.linear_3(h)
        return y
