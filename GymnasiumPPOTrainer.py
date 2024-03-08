import RLFramework.Network
from RLFramework.PPO.PPOTrainer import PPOTrainer
import gymnasium as gym

from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumAgent import GymnasiumAgent
from plotter import  Plotter


class GymnasiumPPOTrainer(PPOTrainer):
    def __init__(self, environment: GymnasiumEnvironment, agent: GymnasiumAgent, value_network: RLFramework.Network,
                 gamma=0.99, batch_size=128, do_train=True, **kwargs):
        super().__init__(policy_net=agent.network, value_net=value_network, environment=environment, agent=agent,
                         gamma=gamma, miniabtch_size=batch_size, **kwargs)
        self.episode = 1

        self.do_train = do_train

        self.plotter = Plotter()
        self.plotter.make_slot(qvalue=0, discount_qvalue=0, obj=0)

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
