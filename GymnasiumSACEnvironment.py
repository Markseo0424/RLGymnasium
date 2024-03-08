from RLFramework.Environment import Environment
import gymnasium as gym


class GymnasiumEnvironment(Environment):
    def __init__(self, env_name: str, discount_factor=0.95, **kwargs):
        super().__init__()
        self.env = gym.make(env_name, **kwargs)
        self.gym_reward = 0
        self.end = False

        self.discount_factor = discount_factor

        self.episode_reward = 0
        self.discount_reward = 0

    def update(self, state, action):
        if state is None:
            observation, info = self.env.reset()
            return observation

        observation, reward, terminated, truncated, info = self.env.step(action[0])

        self.gym_reward = reward
        self.end = terminated or truncated

        if self.end:
            return None

        return observation

    def reward(self, state, action, next_state):
        # print(self.gym_reward)
        self.episode_reward += self.gym_reward
        self.discount_reward = self.discount_reward * self.discount_factor + self.gym_reward

        if state is not None:
            return self.gym_reward# + abs(state[1]) * 10
        else:
            return self.gym_reward

    def reset_params(self):
        self.gym_reward = 0
        self.end = False

        self.episode_reward = 0
        self.discount_reward = 0
