import RLFramework.Network
from RLFramework.SAC.SACTrainer import SACTrainer
import gymnasium as gym
import numpy as np

from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumAgent import GymnasiumAgent
from plotter import Plotter


class GymnasiumSACTrainer(SACTrainer):
    def __init__(self, environment: GymnasiumEnvironment, agent: GymnasiumAgent, value_network: RLFramework.Network,
                 q_nets: tuple[RLFramework.Network, RLFramework.Network],
                 gamma=0.99, batch_size=128, do_train=True, **kwargs):
        super().__init__(policy_net=agent.network, q_nets=q_nets, value_net=value_network,
                         environment=environment, agent=agent,
                         gamma=gamma, batch_size=batch_size, **kwargs)
        self.episode = 1

        self.do_train = do_train

        self.plotter = Plotter()
        self.plotter.make_slot(qvalue=0, discount_qvalue=0, actor_loss=0, critic_loss=0)

    def train(self, state, action, reward, next_state):
        _,__,___ = super().train(state, action, reward, next_state)
        # print(_,__,___)
        # self.plotter.update(actor_loss=actor_loss, critic_loss=critic_loss)

    def reset(self):
        print(f"timestep : {self.timestep}")
        super().reset()

    def check_reset(self):
        return self.environment.end

    def check_train(self):
        return super().check_train() and self.do_train

    def reset_params(self):
        print(f"episode done : {self.episode}")
        self.episode += 1
        self.plotter.step()

    def memory(self):
        super().memory()
        self.plotter.update(qvalue=self.environment.episode_reward, discount_qvalue=self.environment.discount_reward)
