import torch
from torch.optim.adam import Adam
import matplotlib.pyplot as plt
from GymnasiumAgent import GymnasiumAgent
from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumTrainer import GymnasiumTrainer
from CartPoleNetwork import CartPoleNetwork

gamma = 0.995
epsilon = 1
lr = 1e-4
batch_size = 128
epsilon_decay = 0.995
min_epsilon = 0.01
do_train = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = CartPoleNetwork().to(device)
optim = Adam(network.parameters(), lr=lr)
network.set_optimizer(optim)
# network.load_state_dict(torch.load("./CartPole_10.pth"))

environment = GymnasiumEnvironment("CartPole-v1", discount_factor=gamma)#, render_mode="human")
agent = GymnasiumAgent(environment.env.action_space, network, epsilon=epsilon)
trainer = GymnasiumTrainer(environment, agent, gamma=gamma, batch_size=batch_size,
                           epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, do_train=do_train)

trainer.plotter.enable_realtime_plot(qvalue=1, discount_qvalue=1, ma_window_size=10, figsize=(6, 3))

try:
    while True:
        trainer.step()

except KeyboardInterrupt:
    print("terminated")

trainer.plotter.set_hyperparams(
    gamma=gamma,
    lr=lr,
    batch_size=batch_size,
    epsilon_decay=epsilon_decay,
    min_epsilon=min_epsilon
)
if do_train:
    trainer.plotter.save("CartPole_12.json")
    torch.save(network.state_dict(), "log/CartPole_12.pth")

environment.env.close()
