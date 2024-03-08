import RLFramework.Network
from RLFramework.DDPG.DDPGTrainer import DDPGTrainer
import gymnasium as gym
import numpy as np

from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumAgent import GymnasiumAgent
from plotter import Plotter


class GymnasiumDDPGTrainer(DDPGTrainer):
    def __init__(self, environment: GymnasiumEnvironment, agent: GymnasiumAgent, value_network: RLFramework.Network,
                 gamma=0.99, batch_size=128, do_train=True, **kwargs):
        super().__init__(policy_net=agent.network, q_net=value_network, environment=environment, agent=agent,
                         gamma=gamma, batch_size=batch_size, slot_weights={"upper": 1, "plus": 2, "minus": 2}, **kwargs)
        self.episode = 1

        self.do_train = do_train

        self.plotter = Plotter()
        self.plotter.make_slot(qvalue=0, discount_qvalue=0, actor_loss=0, critic_loss=0)

    def choose_slot(self, state, action, reward, next_state):
        if reward < -50:
            return "upper"
        else:
            if reward > 0:#np.sum(action) > 0:
                return "plus"
            else:
                return "minus"

    def train(self, state, action, reward, next_state):
        actor_loss, critic_loss = super().train(state, action, reward, next_state)
        self.plotter.update(actor_loss=actor_loss, critic_loss=critic_loss)

    def reset(self):
        super().reset()
        self.agent.random_process.reset()

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
