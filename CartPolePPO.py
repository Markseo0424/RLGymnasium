import torch
from torch.optim.adam import Adam
import matplotlib.pyplot as plt
from GymnasiumStochasticAgent import GymnasiumAgent
from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumPPOTrainer import GymnasiumPPOTrainer
from CartPoleNetwork import CartPolePolicyNetwork, CartPoleValueNetwork

gamma = 0.99
lr = 2e-4
batch_size = 128
do_train = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = CartPolePolicyNetwork().to(device)
value_network = CartPoleValueNetwork().to(device)

optim = Adam(network.parameters(), lr=lr)
value_optim = Adam(value_network.parameters(), lr=lr)

network.set_optimizer(optim)
value_network.set_optimizer(value_optim)
# network.load_state_dict(torch.load("./CartPole_13_policy.pth"))

environment = GymnasiumEnvironment("CartPole-v1", discount_factor=gamma)#, render_mode="human")
agent = GymnasiumAgent(environment.env.action_space, network)
trainer = GymnasiumPPOTrainer(environment, agent, value_network=value_network, gamma=gamma, batch_size=batch_size,
                              do_train=do_train, verbose="policy", epsilon=0.2, entropy_weight=0.01)

trainer.plotter.enable_realtime_plot(qvalue=1, discount_qvalue=1, ma_window_size=10, figsize=(6, 3))

try:
    while True:
        trainer.step()

except KeyboardInterrupt:
    print("terminated")

trainer.plotter.set_hyperparams(
    method="ppo",
    gamma=gamma,
    lr=lr,
    batch_size=batch_size
)

if do_train:
    trainer.plotter.save("CartPole_17.json")
    torch.save(network.state_dict(), "log/CartPole_17_policy.pth")
    torch.save(value_network.state_dict(), "log/CartPole_17_value.pth")

environment.env.close()
