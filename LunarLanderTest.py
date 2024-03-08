import torch
from torch.optim.adam import Adam
from GymnasiumAgent import GymnasiumAgent
from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumTrainer import GymnasiumTrainer
from LunarLanderNetwork import LunarLanderNetwork

gamma = 0.99
epsilon = 1
lr = 5e-4
batch_size = 256
epsilon_decay = 0.99
min_epsilon = 0.1
do_train = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = LunarLanderNetwork().to(device)
optim = Adam(network.parameters(), lr=lr)
network.set_optimizer(optim)
network.load_state_dict(torch.load("log/LunarLander_1.pth"))

environment = GymnasiumEnvironment("LunarLander-v2", render_mode="human", discount_factor=gamma)
agent = GymnasiumAgent(environment.env.action_space, network, epsilon=epsilon)
trainer = GymnasiumTrainer(environment, agent, gamma=gamma, batch_size=batch_size, do_train=do_train)

trainer.plotter.enable_realtime_plot(qvalue=1, discount_qvalue=1, ma_window_size=100, figsize=(6, 3))
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
    trainer.plotter.save("LunarLander_2.json")
    torch.save(network.state_dict(), "LunarLander_2.pth")

environment.env.close()
