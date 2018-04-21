import matplotlib as mpl
import matplotlib.pyplot as plt


print(mpl.rcParams['agg.path.chunksize'])


def plot_results(Q_updates, right_axis=None, title='Q Updates', right_axis_title='Reward'):
    fig, ax1 = plt.subplots()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax1.set_xlabel('Simulation Iteration')
    ax1.plot(Q_updates, 'b-')
    ax1.set_ylabel('Q-value Difference')
    ax1.set_ylim((0, 0.5))
    ax1.tick_params('y')

    if right_axis is not None:
        ax2 = ax1.twinx()
        ax2.plot(right_axis, 'g-')
        ax2.set_ylabel(right_axis_title)
        ax2.tick_params('y')

    fig.tight_layout()
    plt.title(title)
    plt.show()
