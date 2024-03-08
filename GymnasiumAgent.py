from RLFramework.Agent import Agent
import torch
import gymnasium as gym
import random


class GymnasiumAgent(Agent):
    def __init__(self, action_space, network, epsilon=1):
        super().__init__()
        self.network = network
        self.epsilon = epsilon
        if type(action_space) is gym.spaces.Discrete:
            self.action_space = action_space
            self.ACTIONS = list(range(action_space.n))
        else:
            raise NotImplementedError

    def policy(self, state):
        if state is None:
            return None

        pred = self.network.predict(state)
        # print(pred)

        if random.random() < self.epsilon:
            return self.action_space.sample()

        return torch.argmax(pred).item()

    def reset_params(self):
        pass