import torch
import torch.nn as nn
from RLFramework.Network import Network


class MountainActorNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, 2))
        else:
            unbatched = False
        x = self.model(x)

        if unbatched:
            x = x.reshape(1)

        return x


class MountainCriticNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, 3))
        else:
            unbatched = False

        x = self.model(x)

        if unbatched:
            x = x.reshape(1)

        return x
