import torch
import torch.nn as nn
from RLFramework.Network import Network


class MountainPolicyNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def sample_action(self, policy):
        if len(policy.shape) == 1:
            unbatched = True
            policy = policy.reshape((1, -1))
        else:
            unbatched = False

        dist = torch.distributions.normal.Normal(policy[:, :1], torch.exp(policy[:, 1:] / 2))
        action = dist.sample().to(self.device)
        logprobs = torch.sum(dist.log_prob(action), dim=1)

        if unbatched:
            action = action.reshape(-1)

        return action, logprobs

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, -1))
        else:
            unbatched = False
        x = self.model(x)

        if unbatched:
            x = x.reshape(-1)

        return x


class MountainQNetwork(Network):
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
            x = x.reshape((1, -1))
        else:
            unbatched = False

        x = self.model(x)

        if unbatched:
            x = x.reshape(-1)

        return x


class MountainValueNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 64),
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
            x = x.reshape((1, -1))
        else:
            unbatched = False

        x = self.model(x)

        if unbatched:
            x = x.reshape(-1)

        return x
