#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt

from project1.settings import NSTATES, ACTUAL_STATE_VALUES, OUTPUT_DIR


def plot(x, y, xlab=u'$\lambda$', ylab='ERROR'):
    plt.plot(x, y, label='test')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    # plt.legend()
    plt.show()


def plot_alpha(x, y, ld_vals, xlab=u'$\\alpha$', ylab='ERROR'):
    for i in range(len(y)):
        plt.plot(x, y[i], label=ld_vals[i])
    plt.xlim((-0.05, 0.65))
    plt.ylim((0.05, 0.75))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc=2)
    plt.show()


def plot_val_estimates(file_counter, state_vals, episode, alpha):
    fig = plt.figure()
    x = range(1, NSTATES + 1)
    plt.plot(x, state_vals, label='Estimated')  # [s.v for s in states][1:NSTATES + 1]
    plt.plot(x, ACTUAL_STATE_VALUES, label='Optimal')
    plt.xticks(x, ['A', 'B', 'C', 'D', 'E'])
    plt.title('State value estimates')
    plt.xlim((1, NSTATES))
    plt.ylim((0, 1))
    plt.text(4.2, 0.14, 'T = {}'.format(episode))
    plt.text(4.2, 0.10, u'$\\alpha$ = {}'.format(round(alpha, 4)))
    plt.legend(title='lambda')
    plt.legend(loc=2)
    fig.savefig(os.path.join('output', OUTPUT_DIR, '{}.png'.format(str(file_counter).zfill(4))))
    plt.close()


def plot_value_updates():
    pass


def plot_td_mc_comparison():
    pass
