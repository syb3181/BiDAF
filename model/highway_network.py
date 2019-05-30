import torch.nn as nn


class HighwayNetwork(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Linear(dim, dim, False)
        self.f = nn.Linear(dim, dim, False)
        self.relu = nn.ReLU()

    def forward(self, x):
        gate = self.relu(self.g(x))
        f = self.relu(self.f(x))
        return gate * f + (1 - gate) * x
