import torch
import torch.nn as nn
from RLFramework.Network import Network


class BipedalActorNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(24),
            nn.Linear(24, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 300),
            nn.LeakyReLU(0.2),
            nn.Linear(300, 4),
            nn.Tanh()
        )

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, 24))
        else:
            unbatched = False

        x = self.model(x)

        if unbatched:
            x = x.reshape(4)

        return x


class BipedalCriticNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(28),
            nn.Linear(28, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 300),
            nn.LeakyReLU(0.2),
            nn.Linear(300, 1)
        )

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, 28))
        else:
            unbatched = False

        x = self.model(x)

        if unbatched:
            x = x.reshape(1)

        return x
