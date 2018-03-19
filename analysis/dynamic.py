import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt

from . dynamic_plot import exchanges, moves


def plot_moves(data, number):

    bkp = data.data[number]

    agent_type = np.array(
        [0, ] * bkp.parameters.x0 + [1, ] * bkp.parameters.x1 + [2, ] * bkp.parameters.x2)

    n = bkp.parameters.x0 + bkp.parameters.x1 + bkp.parameters.x2

    a = bkp.agent_maps.copy()
    for i in range(n):
        a[a == i] = agent_type[i]

    if bkp.parameters.stride > 0:
        moves.plot(data=a)

    else:
        color_list = ['white', 'C0', 'C1', 'C2']
        color_map = matplotlib.colors.ListedColormap(color_list)

        fig = plt.figure(figsize=(10, 10), facecolor='white', dpi=72)
        ax = fig.add_subplot(111)

        ax.imshow(
            a[0], interpolation='none', aspect='auto', origin='upper',
            cmap=color_map, vmin=-1, vmax=len(color_list) - 1)

        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.show()


def plot_exchanges(data, number):
    print(data.data[number].exchange_maps.shape[0])
    exchanges.plot(data.data[number].exchange_maps, mode=exchanges.Mode.ON_KEY_PRESS)

