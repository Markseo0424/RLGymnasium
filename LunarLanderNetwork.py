import torch.nn as nn
from RLFramework.Network import Network


class LunarLanderNetwork(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, 8))
        else:
            unbatched = False

        x = self.model(x)

        if unbatched:
            x = x.reshape(4)

        return x
