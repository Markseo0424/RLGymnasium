import matplotlib.pyplot as plt
import json


class Plotter:
    def __init__(self):
        self.memory = {}
        self.cache = {}
        self.settings = {}

        self.realtime_plot = False
        self.realtime_plot_slots = {}
        self.realtime_plot_colors = None
        self.ax = None
        self.ma_window_size = 1
        self.fig = None

    def make_slot(self, **kwargs):
        for key in kwargs.keys():
            if key in self.memory.keys():
                continue
            else:
                self.memory[key] = [kwargs[key]]

    def update(self, **kwargs):
        for key in kwargs.keys():
            assert key in self.memory.keys()
            self.cache[key] = kwargs[key]

    def enable_realtime_plot(self, ax=None, ma_window_size=1, figsize=(15, 7), **slots):
        self.realtime_plot = True
        self.realtime_plot_slots = slots

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

        self.ax = ax
        self.ma_window_size = ma_window_size

        labels, colors = self.weighted_plot(ax=self.ax, ma_window_size=self.ma_window_size, length=100, **self.realtime_plot_slots)
        self.realtime_plot_colors = dict(zip(labels,colors))
        plt.legend(loc='center left')
        plt.pause(0.00001)

        return ax

    def step(self):
        for key in self.memory.keys():
            if key in self.cache.keys():
                self.memory[key].append(self.cache[key])
            else:
                self.memory[key].append(self.memory[key][-1])

        if self.realtime_plot:
            self.ax.cla()
            self.weighted_plot(ax=self.ax, ma_window_size=self.ma_window_size, length=100, **self.realtime_plot_slots)
            plt.pause(0.00001)

    def set_hyperparams(self, **kwargs):
        self.settings = kwargs

    def save(self, fp):
        jsonData = {
            "settings": self.settings,
            "memory": self.memory
        }
        with open(fp, 'w') as f:
            json.dump(jsonData, f, indent=2)

    def load(self, fp):
        with open(fp, 'r') as f:
            jsonData = json.load(f)

        if "memory" in jsonData.keys():
            self.memory = jsonData["memory"]
            self.settings = jsonData["settings"]
        else:
            self.memory = jsonData

    def concat(self, fp):
        with open(fp, 'r') as f:
            jsonData = json.load(f)

        if "memory" in jsonData.keys():
            memory = jsonData["memory"]
        else:
            memory = jsonData

        for key in memory.keys():
            if key in self.memory.keys():
                self.memory[key] = self.memory[key] + memory[key]

    def merge(self, fp, prefix):
        with open(fp, 'r') as f:
            jsonData = json.load(f)

        if "memory" in jsonData.keys():
            memory = jsonData["memory"]
        else:
            memory = jsonData

        for key in memory.keys():
            self.memory[prefix + "." + key] = memory[key]

    def get_moving_average(self, data, window, mult=1):
        res = []
        for i in range(len(data) - window + 1):
            res.append(mult * sum(data[i:i + window]) / window)
        return res

    def plot(self, *slots, ax=None, ma_window_size=None, prefix=None):
        kwargs = {}

        for s in slots:
            kwargs[s] = 1

        return self.weighted_plot(ma_window_size=ma_window_size, ax=ax, prefix=prefix, **kwargs)

    def weighted_plot(self, ma_window_size=None, ax=None, prefix=None, length=-1, **slots):
        multi_graph = ax is not None
        labels = []
        colors = []

        if not multi_graph:
            ax = plt

        for _slot in slots.keys():
            if prefix is not None:
                slot = prefix + "." + _slot
            else:
                slot = _slot

            assert slot in self.memory.keys()

            if length != -1 and len(self.memory[slot]) > length:
                value = self.memory[slot][-length:]
            else:
                value = self.memory[slot]

            if ma_window_size is not None:
                value = self.get_moving_average(value, ma_window_size, slots[_slot])
            else:
                value = value

            label = slot + f" x {slots[_slot]}" if slots[_slot] != 1 else slot

            if self.realtime_plot and self.realtime_plot_colors is not None:
                line = ax.plot(value, label=label, color=self.realtime_plot_colors[label])
            else:
                line = ax.plot(value, label=label)
            labels.append(label)
            colors.append(line[-1].get_color())

        if not multi_graph:
            plt.legend()
            plt.show()

        return labels, colors
