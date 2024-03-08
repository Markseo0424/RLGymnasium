import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(
            next_states), np.array(dones, dtype=np.uint8)

    def __len__(self):
        return len(self.buffer)


class SoftActorCritic:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, alpha=0.2, tau=0.005):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr/10)

        self.critic1 = Critic(state_dim, action_dim)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = Critic(state_dim, action_dim)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state).squeeze(0)
        return action.numpy()

    def update(self, replay_buffer, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        with torch.no_grad():
            next_action = self.actor_target(next_state_batch)
            noise = torch.clamp(torch.normal(0, 0.2, size=next_action.size()), -0.5, 0.5)
            next_action += noise
            next_action = torch.clamp(next_action, -1, 1)

            target_Q1 = self.critic1_target(next_state_batch, next_action)
            target_Q2 = self.critic2_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1 - done_batch) * self.gamma * (target_Q - torch.exp(self.log_alpha) * target_Q)

        current_Q1 = self.critic1(state_batch, action_batch)
        current_Q2 = self.critic2(state_batch, action_batch)

        critic1_loss = F.mse_loss(current_Q1, target_Q.detach())
        critic2_loss = F.mse_loss(current_Q2, target_Q.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        sampled_actions = self.actor(state_batch)
        actor_loss = (torch.exp(self.log_alpha) * self.compute_entropy(sampled_actions) - self.critic1(state_batch,
                                                                                        sampled_actions)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (self.compute_entropy(sampled_actions) + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        print(alpha_loss)

    def compute_entropy(self, actions):
        log_probs = torch.log(torch.clamp(actions, 1e-6, 1.0))
        entropy = -torch.sum(actions * log_probs, dim=-1)
        return entropy.mean()


# Create environment
env = gym.make('Pendulum-v1', render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Initialize agent
agent = SoftActorCritic(state_dim, action_dim)

# Initialize replay buffer
replay_buffer = ReplayBuffer(capacity=10000)

# Training loop
num_episodes = 1000
batch_size = 64

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        # Select action
        action = agent.select_action(state)
        print(action)

        # Take action in the environment
        next_state, reward, done, _, __ = env.step(action)

        # Store transition in the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        # Perform SAC update
        if len(replay_buffer) > batch_size:
            agent.update(replay_buffer, batch_size)

        if done:
            break

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

env.close()