import torch
import torch.nn as nn
from RLFramework.Network import Network


class PendulumPolicyNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Linear(3, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def sample_action(self, policy, test=False):
        if len(policy.shape) == 1:
            unbatched = True
            policy = policy.reshape((1, -1))
        else:
            unbatched = False

        if not test:
            dist = torch.distributions.normal.Normal(policy[:, :1], torch.exp(policy[:, 1:]))
            action = dist.rsample()

            logprobs = dist.log_prob(action)
            action = torch.tanh(action)

            logprobs -= torch.log(2 * (1 - action.pow(2)) + 1e-6)
            logprobs = torch.sum(logprobs, dim=1, keepdim=True)
            action = action * 2
        else:
            action = policy[:, :1]
            action = torch.tanh(action)

            action = action * 2

            logprobs = None



        #print(f"pocliy : {policy}")
        #print(f"action : {action}, logprob : {logprobs}")

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


class PendulumQNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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


class PendulumValueNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
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
