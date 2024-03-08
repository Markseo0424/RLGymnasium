from RLFramework.Agent import Agent
import torch
import numpy as np
import gymnasium as gym
import random


class GymnasiumAgent(Agent):
    def __init__(self, action_space, network):
        super().__init__()
        self.network = network

        if type(action_space) is gym.spaces.Discrete:
            self.action_space = action_space
            self.ACTIONS = list(range(action_space.n))
        else:
            raise NotImplementedError

    def policy(self, state):
        if state is None:
            return None

        pred = self.network.predict(state)

        p = pred.detach().cpu().numpy()

        return np.random.choice(a=len(self.ACTIONS), p=p)

    def reset_params(self):
        pass