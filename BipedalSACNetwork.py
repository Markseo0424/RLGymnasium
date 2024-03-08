import torch
import torch.nn as nn
from RLFramework.Network import Network


class BipedalPolicyNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(24),
            nn.Linear(24, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 300),
            nn.LeakyReLU(0.2),
            nn.Linear(300, 8)
        )

        self.mean_activation = nn.Tanh()
        self.logvar_activation = None

    def sample_action(self, policy, test=False):
        if len(policy.shape) == 1:
            unbatched = True
            policy = policy.reshape((1, -1))
        else:
            unbatched = False

        if not test:
            dist = torch.distributions.normal.Normal(policy[:, :4], torch.exp(torch.clamp(policy[:, 4:], -20, 20)))
            action = dist.rsample()

            logprobs = torch.log(torch.exp(dist.log_prob(action)) + 1e-9)
            action = torch.tanh(action)
            print(f"action before scale: {action}, logprobs before scale: {logprobs}")

            logprobs -= torch.log((1 - action.pow(2)) + 1e-9)
            logprobs = torch.sum(logprobs, dim=1, keepdim=True)
            action = action
        else:
            action = policy[:, :4]
            action = torch.tanh(action)

            action = action

            logprobs = None

        # print(f"pocliy : {policy}")
        # print(f"action : {action}, logprob : {logprobs}")

        if unbatched:
            action = action.reshape(-1)

        return action, logprobs

    def forward(self, x):
        if len(x.shape) == 1:
            unbatched = True
            x = x.reshape((1, 24))
        else:
            unbatched = False

        x = self.model(x)

        if self.mean_activation is not None:
            mean = self.mean_activation(x[:, :4])
        else:
            mean = x[:, :4]

        if self.logvar_activation is not None:
            logvar = self.logvar_activation(x[:, 4:])
        else:
            logvar = x[:, 4:]

        x = torch.cat([mean, logvar], dim=1)

        if unbatched:
            x = x.reshape(8)

        return x


class BipedalQNetwork(Network):
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


class BipedalValueNetwork(Network):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(24),
            nn.Linear(24, 400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 300),
            nn.LeakyReLU(0.2),
            nn.Linear(300, 1)
        )

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
