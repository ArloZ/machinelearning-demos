# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def plot_scatter(x, y, size=None, color=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, size, color)
    if show:
        plt.show()


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    y = [1, 5, 3, 2, 4]
    plot_scatter(x, y)
