import matplotlib.pyplot as plt


def plot_results(Q_updates, rewards=None, title='Q Updates'):
    fig, ax1 = plt.subplots()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax1.set_xlabel('Simulation Iteration')
    ax1.plot(Q_updates, 'b-')
    ax1.set_ylabel('Q-value Difference')
    ax1.tick_params('y')

    if rewards is not None:
        ax2 = ax1.twinx()
        ax2.plot(rewards, 'g-')
        ax2.set_ylabel('Reward')
        ax2.tick_params('y')

    fig.tight_layout()
    plt.title(title)
    plt.show()
