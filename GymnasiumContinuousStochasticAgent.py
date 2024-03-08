from RLFramework.Agent import Agent
import torch
import numpy as np
import gymnasium as gym
import random
from OUProcess import OUProcess


class GymnasiumAgent(Agent):
    def __init__(self, action_space, network, test=False):
        super().__init__()
        self.network = network
        self.test = test

        if type(action_space) is gym.spaces.Discrete:
            self.action_space = action_space
            self.isContinuous = False
            self.ACTIONS = list(range(action_space.n))
        else:
            self.isContinuous = True
            self.action_space = action_space

    def policy(self, state):
        if state is None:
            return None
        print(f"state: {state}")
        pred = self.network.predict(state)
        print(f"pred: {pred}")
        # print(pred)
        if self.isContinuous:
            if not self.test:
                action, logprob = self.network.sample_action(pred)
                print(f"action: {action}, logprob: {logprob}")
                return action.cpu().detach().numpy(), logprob.cpu().detach().numpy()
            else:
                action, logprob = self.network.sample_action(pred, True)
                return action.cpu().detach().numpy(), 1


        else:
            if random.random() < self.epsilon:
                return self.action_space.sample()

            return torch.argmax(pred).item()

    def reset_params(self):
        pass