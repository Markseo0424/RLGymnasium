from RLFramework.Agent import Agent
import torch
import numpy as np
import gymnasium as gym
import random
from OUProcess import OUProcess


class GymnasiumAgent(Agent):
    def __init__(self, action_space, network, epsilon=1, sigma=0.5):
        super().__init__()
        self.network = network
        self.epsilon = epsilon

        if type(action_space) is gym.spaces.Discrete:
            self.action_space = action_space
            self.isContinuous = False
            self.ACTIONS = list(range(action_space.n))
        else:
            self.isContinuous = True
            self.action_space = action_space
            self.random_process = OUProcess(size=self.action_space.shape, sigma=sigma)

    def policy(self, state):
        if state is None:
            return None

        pred = self.network.predict(state)

        if self.isContinuous:
            pred = pred.cpu().detach().numpy()
            sample = self.epsilon * self.random_process.sample()

            # print(f"{pred} + {sample}")

            return np.clip(pred + sample, a_min=-1, a_max=1)

        else:
            if random.random() < self.epsilon:
                return self.action_space.sample()

            return torch.argmax(pred).item()

    def reset_params(self):
        pass