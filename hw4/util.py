import os
import sys

import matplotlib.pyplot as plt


def save_txt(env, output_dir, episode_dir, iteration):
    stdout = sys.stdout
    new_file = os.path.join('output', output_dir, episode_dir, '{}.txt'.format(str(iteration).zfill(4)))
    sys.stdout = open(new_file, 'w')
    env.render()
    sys.stdout = stdout


def plot_results(Q_updates, rewards):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episode')
    ax1.plot(Q_updates, 'b-')
    ax1.set_ylabel('|$\Delta$Q|')
    ax1.tick_params('y')

    ax2 = ax1.twinx()
    ax2.plot(rewards, 'g-')
    ax2.set_ylabel('Reward')
    ax2.tick_params('y')

    fig.tight_layout()
    plt.show()
