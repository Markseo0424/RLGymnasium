import torch.nn as nn
from RLFramework.Network import Network


class CartPoleNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, 4))
        else:
            unbatched = False

        x = self.model(x)

        if unbatched:
            x = x.reshape(2)

        return x


class CartPolePolicyNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, 4))
        else:
            unbatched = False

        x = self.model(x)
        x = nn.Softmax(dim=1)(x)

        if unbatched:
            x = x.reshape(2)

        return x

class CartPoleValueNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, 4))
        else:
            unbatched = False

        x = self.model(x)

        if unbatched:
            x = x.reshape(1)

        return x
