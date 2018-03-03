import matplotlib.pyplot as plt


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
    plt.title('Q Updates and Rewards')
    plt.show()
