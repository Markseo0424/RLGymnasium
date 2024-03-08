import matplotlib.pyplot as plt
from plotter import Plotter


plot_kwargs = {"qvalue": 1, "discount_qvalue": 1}

plotter = Plotter()

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(1, 1, 1)

plotter.load("log/InvPendulum_SAC_1.json")
labels, colors = plotter.weighted_plot(**plot_kwargs, ma_window_size=10, ax=ax)

fig.legend(labels, loc="center right")
plt.suptitle("SAC train graph")

plt.subplots_adjust(top=0.95, bottom=0.05, right=0.85, left=0.05)

plt.savefig("plot/SAC_InvPendulum_1_graph.png")
plt.show()
