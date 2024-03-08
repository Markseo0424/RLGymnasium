import numpy as np


class OUProcess:
    def __init__(self, size, seed=0, mu=0., theta=0.15, sigma=0.5):
        self.size = size

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        np.random.seed(seed)

        self.state = None
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.normal(0, 1, size=self.size)
        self.state += dx
        return self.state
