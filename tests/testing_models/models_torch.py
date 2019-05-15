import torch
import torch.nn
import torch.nn.functional as F
import torchvision.models as models


class MLP(torch.nn.Module):
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 512)
        self.linear_2 = torch.nn.Linear(512, 256)
        self.linear_3 = torch.nn.Linear(256, 128)
        self.linear_4 = torch.nn.Linear(128, 64)
        self.linear_5 = torch.nn.Linear(64, 32)
        self.linear_6 = torch.nn.Linear(32, 16)
        self.linear_7 = torch.nn.Linear(16, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        h = F.relu(self.linear_2(h))
        h = F.relu(self.linear_3(h))
        h = F.relu(self.linear_4(h))
        h = F.relu(self.linear_5(h))
        h = F.relu(self.linear_6(h))
        y = self.linear_7(h)
        return y