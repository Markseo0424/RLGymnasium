import torch
from torch.optim.adam import Adam
from GymnasiumStochasticAgent import GymnasiumAgent
from GymnasiumEnvironment import GymnasiumEnvironment
from GymnasiumTRPOTrainer import GymnasiumTRPOTrainer
from LunarLanderPolicyNetwork import LunarLanderPolicyNetwork
from LunarLanderValueNetwork import LunarLanderValueNetwork

gamma = 0.99
epsilon = 1
lr = 5e-4
batch_size = 10000
epsilon_decay = 0.99
min_epsilon = 0.1
do_train = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_network = LunarLanderPolicyNetwork().to(device)
value_network = LunarLanderValueNetwork().to(device)
optim = Adam(value_network.parameters(), lr=lr)
value_network.set_optimizer(optim)
#value_network.load_state_dict(torch.load("./LunarLander_1.pth"))

environment = GymnasiumEnvironment("LunarLander-v2", render_mode="human", discount_factor=gamma)
agent = GymnasiumAgent(environment.env.action_space, policy_network)
trainer = GymnasiumTRPOTrainer(environment, agent, gamma=gamma, batch_size=batch_size, value_network=value_network,
                               do_train=do_train, delta=0.01, CG_iter=50, damping=0.1, alpha=0.5)

trainer.plotter.enable_realtime_plot(qvalue=1, discount_qvalue=1, obj=50, ma_window_size=100, figsize=(6, 3))
try:
    while True:
        trainer.step()
except:
    print("terminated")

trainer.plotter.set_hyperparams(
    gamma=gamma,
    lr=lr,
    batch_size=batch_size,
    epsilon_decay=epsilon_decay,
    min_epsilon=min_epsilon
)
if do_train:
    trainer.plotter.save("LunarLander_3.json")
    torch.save(policy_network.state_dict(), "log/LunarLander_policy_3.pth")
    torch.save(value_network.state_dict(), "log/LunarLander_value_3.pth")

environment.env.close()