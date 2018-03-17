import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def plot(x, key=None, title='Rewards per episode', xlab='', ylab='', multi=False, save=True):
    if multi:
        for _ in x:
            plt.plot(_)
    else:
        plt.plot(x)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if key is not None:
        plt.legend(key)
    if save:
        plt.savefig('{}.png'.format(datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')))
    else:
        try:
            plt.show()
        except:
            plt.savefig('{}.png'.format(datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')))
    plt.close()


if __name__ == '__main__':
    x = np.arange(1000)
    plt.plot(x, np.power(0.995, x))
    plt.plot(x, np.power(0.99, x))
    plt.plot(x, np.power(0.9, x))
    plt.plot(x, [1.0 if i < 250 else 0 for i in range(1000)])
    plt.title('Epsilon functions')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.show()
