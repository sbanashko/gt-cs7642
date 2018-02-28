import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def get_samples(Q):
    return [{
        'expected': -11.374402515,
        'actual': Q[462, 4]
    }, {
        'expected': 4.348907,
        'actual': Q[398, 3]
    }, {
        'expected': -0.5856821173,
        'actual': Q[253, 0]
    }, {
        'expected': 9.683,
        'actual': Q[377, 1]
    }, {
        'expected': -12.8232660372,
        'actual': Q[83, 5]
    }]


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
    plt.title('Q Updates and Rewards')
    plt.show()


def validate_results(Q):
    problems = get_samples(Q)
    for i, p in enumerate(problems):
        result = '✔' if np.isclose(p['expected'], p['actual']) else 'X'
        print()
        print('{} Problem {}'.format(result, i))
        print('Expected: {}'.format(p['expected']))
        print('Actual: {}'.format(p['actual']))

    print()


def show_hw_answers(Q):
    print()
    print('*' * 80)
    print('Homework Answers')
    print('=' * 80)
    print('Problem 1: ', Q[423, 0])
    print('Problem 2: ', Q[178, 2])
    print('Problem 3: ', Q[306, 2])
    print('Problem 4: ', Q[487, 2])
    print('Problem 5: ', Q[253, 5])
    print('Problem 6: ', Q[194, 0])
    print('Problem 7: ', Q[359, 1])
    print('Problem 8: ', Q[474, 5])
    print('Problem 9: ', Q[142, 2])
    print('Problem 10: ', Q[448, 0])
    print('*' * 80)
    print()
