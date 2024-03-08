import torch
from torch.optim.adam import Adam
import matplotlib.pyplot as plt
from GymnasiumContinuousStochasticAgent import GymnasiumAgent
from GymnasiumSACEnvironment import GymnasiumEnvironment
from GymnasiumSACTrainer import GymnasiumSACTrainer
from BipedalSACNetwork import *

gamma = 0.99
train_epsilon = 1
test_epsilon = 0
lr = 3e-4
batch_size = 64
do_train = True
test = True
show = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = BipedalPolicyNetwork().to(device)
value_net = BipedalValueNetwork().to(device)
q_net_1 = BipedalQNetwork().to(device)
q_net_2 = BipedalQNetwork().to(device)

policy_net.set_optimizer(Adam(policy_net.parameters(), lr=0.01))
value_net.set_optimizer(Adam(value_net.parameters(), lr=0.01))
q_net_1.set_optimizer(Adam(q_net_1.parameters(), lr=0.01))
q_net_2.set_optimizer(Adam(q_net_2.parameters(), lr=0.01))

if do_train:
    environment = GymnasiumEnvironment("BipedalWalker-v3", discount_factor=gamma, render_mode="human" if show else None)
    agent = GymnasiumAgent(environment.env.action_space, policy_net)
    trainer = GymnasiumSACTrainer(environment, agent, q_nets=(q_net_1, q_net_2), value_network=value_net, gamma=gamma,
                                  batch_size=batch_size, start_train_step=10000, train_freq=1, alpha=1, tau=0.005,
                                  do_train=True, buffer_len=100000)

    trainer.plotter.enable_realtime_plot(qvalue=1, discount_qvalue=1, ma_window_size=10, figsize=(6, 3))

    try:
        while True:
            trainer.step()

    except KeyboardInterrupt:
        print("terminated")

    trainer.plotter.set_hyperparams(
        method="sac",
        gamma=gamma,
        lr=lr,
        batch_size=batch_size
    )
    trainer.plotter.save("log/Bipedal_SAC_1.json")
    torch.save(policy_net.state_dict(), "log/Bipedal_policy_SAC_1.pth")
    torch.save(value_net.state_dict(), "log/Bipedal_value_SAC_1.pth")
    torch.save(q_net_1.state_dict(), "log/Bipedal_q1_SAC_1.pth")
    torch.save(q_net_2.state_dict(), "log/Bipedal_q2_SAC_1.pth")

    environment.env.close()

if test:
    # test
    policy_net.load_state_dict(torch.load("log/Bipedal_policy_SAC_1.pth"))
    value_net.load_state_dict(torch.load("log/Bipedal_value_SAC_1.pth"))
    q_net_1.load_state_dict(torch.load("log/Bipedal_q1_SAC_1.pth"))
    q_net_2.load_state_dict(torch.load("log/Bipedal_q2_SAC_1.pth"))

    environment = GymnasiumEnvironment("BipedalWalker-v3", discount_factor=gamma, render_mode="human")
    agent = GymnasiumAgent(environment.env.action_space, policy_net, test=True)
    trainer = GymnasiumSACTrainer(environment, agent, q_nets=(q_net_1, q_net_2), value_network=value_net, gamma=gamma,
                                  batch_size=batch_size, start_train_step=1000, train_freq=16, tau=0.1, do_train=False,
                                  buffer_len=100000)

    try:
        while True:
            trainer.step()

    except KeyboardInterrupt:
        print("terminated")

    environment.env.close()
