from RLFramework.DQN.DQNTrainer import DQNTrainer
import gymnasium as gym

from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumAgent import GymnasiumAgent
from plotter import  Plotter


class GymnasiumTrainer(DQNTrainer):
    def __init__(self, environment: GymnasiumEnvironment, agent: GymnasiumAgent,
                 gamma=0.99, batch_size=128, epsilon_decay=0.995, min_epsilon=0.1, do_train=True):
        super().__init__(agent.network, environment, agent,
                         gamma=gamma, batch_size=batch_size)
        self.episode = 1

        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.do_train = do_train

        if not self.do_train:
            self.agent.epsilon = 0

        self.plotter = Plotter()
        self.plotter.make_slot(qvalue=0, discount_qvalue=0)

    def check_reset(self):
        return self.environment.end

    def check_train(self):
        return super().check_train() and self.do_train

    def reset_params(self):
        print(f"episode done : {self.episode}")
        self.episode += 1
        self.agent.epsilon = max(self.agent.epsilon * self.epsilon_decay, self.min_epsilon) if self.do_train else 0
        self.plotter.step()

    def memory(self):
        super().memory()
        self.plotter.update(qvalue=self.environment.episode_reward, discount_qvalue=self.environment.discount_reward)
