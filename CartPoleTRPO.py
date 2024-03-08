import torch
from torch.optim.adam import Adam
import matplotlib.pyplot as plt
from GymnasiumStochasticAgent import GymnasiumAgent
from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumTRPOTrainer import GymnasiumTRPOTrainer
from CartPoleNetwork import CartPolePolicyNetwork, CartPoleValueNetwork

gamma = 0.99
lr = 3e-4
batch_size = 1024
do_train = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = CartPolePolicyNetwork().to(device)
value_network = CartPoleValueNetwork().to(device)
optim = Adam(value_network.parameters(), lr=lr)
value_network.set_optimizer(optim)
# network.load_state_dict(torch.load("./CartPole_13_policy.pth"))

environment = GymnasiumEnvironment("CartPole-v1", discount_factor=gamma, render_mode="human")
agent = GymnasiumAgent(environment.env.action_space, network)
trainer = GymnasiumTRPOTrainer(environment, agent, value_network=value_network, gamma=gamma, batch_size=batch_size,
                               do_train=do_train, verbose="policy,cg")

trainer.plotter.enable_realtime_plot(qvalue=1, discount_qvalue=1, ma_window_size=10, figsize=(6, 3))

try:
    while True:
        trainer.step()

except KeyboardInterrupt:
    print("terminated")

trainer.plotter.set_hyperparams(
    gamma=gamma,
    lr=lr,
    batch_size=batch_size
)
if do_train:
    trainer.plotter.save("CartPole_14.json")
    torch.save(network.state_dict(), "log/CartPole_14_policy.pth")
    torch.save(value_network.state_dict(), "log/CartPole_14_value.pth")

environment.env.close()