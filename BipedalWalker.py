import torch
from torch.optim.adam import Adam
import matplotlib.pyplot as plt
from GymnasiumDeterministicAgent import GymnasiumAgent
from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumDDPGTrainer import GymnasiumDDPGTrainer
from BipedalNetwork import *

gamma = 0.99
train_epsilon = 1
test_epsilon = 0
sigma = 0.2
tau = 0.001
train_freq = 16

actor_lr = 1e-4
critic_lr = 1e-3

batch_size = 128
do_train = True
test = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

critic_network = BipedalCriticNetwork().to(device)
actor_network = BipedalActorNetwork().to(device)

critic_optim = Adam(critic_network.parameters(), lr=critic_lr)
actor_optim = Adam(actor_network.parameters(), lr=actor_lr)

critic_network.set_optimizer(critic_optim)
actor_network.set_optimizer(actor_optim)

if do_train:
    environment = GymnasiumEnvironment("BipedalWalker-v3", discount_factor=gamma)#, render_mode="human")
    agent = GymnasiumAgent(environment.env.action_space, actor_network, epsilon=train_epsilon, sigma=sigma)
    trainer = GymnasiumDDPGTrainer(environment, agent, value_network=critic_network, gamma=gamma, batch_size=batch_size,
                                   start_train_step=1000, train_freq=train_freq, tau=tau, do_train=True, buffer_len=100000)

    trainer.plotter.enable_realtime_plot(qvalue=1, discount_qvalue=1, ma_window_size=10, figsize=(6, 3))

    try:
        while True:
            trainer.step()

    except KeyboardInterrupt:
        print("terminated")

    trainer.plotter.set_hyperparams(
        method="ddpg",
        gamma=gamma,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        batch_size=batch_size,
        tau=tau,
        sigma=sigma,
        train_freq=train_freq
    )
    trainer.plotter.save("Bipedal_1.json")
    torch.save(critic_network.state_dict(), "log/Bipedal_critic_1.pth")
    torch.save(actor_network.state_dict(), "log/Bipedal_actor_1.pth")

    environment.env.close()

if test:
    # test
    actor_network.load_state_dict(torch.load("log/Bipedal_actor_1.pth"))
    critic_network.load_state_dict(torch.load("log/Bipedal_critic_1.pth"))

    environment = GymnasiumEnvironment("BipedalWalker-v3", discount_factor=gamma, render_mode="human")
    agent = GymnasiumAgent(environment.env.action_space, actor_network, epsilon=test_epsilon)
    trainer = GymnasiumDDPGTrainer(environment, agent, value_network=critic_network, gamma=gamma, do_train=False)

    try:
        while True:
            trainer.step()

    except KeyboardInterrupt:
        print("terminated")

    environment.env.close()
