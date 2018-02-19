#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt

from project1.settings import NSTATES, ACTUAL_STATE_VALUES, OUTPUT_DIR, EX62_T_VALS


def plot(x, y, xlab=u'$\lambda$', ylab='ERROR'):
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    # plt.legend()

    # Temporary curve-fitting (https://plot.ly/matplotlib/polynomial-fits/)

    # # calculate polynomial
    # z = np.polyfit(x, y, 3)
    # fn = np.poly1d(z)

    plt.plot(x, y)
    # plt.plot(x, fn(x))

    plt.show()


def plot_alpha(x, y, ld_vals, xlab=u'$\\alpha$', ylab='ERROR', legend=True, file_counter=0):
    # fig = plt.figure()
    for i in range(len(y)):
        plt.plot(x, y[i], label=ld_vals[i])
    plt.xlim((-0.05, 0.65))
    plt.ylim((0.05, 0.75))
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if legend:
        plt.legend(loc=2)
    # fig.savefig(os.path.join('output', OUTPUT_DIR, '{}.png'.format(str(file_counter).zfill(4))))
    # plt.close()
    plt.show()


def plot_val_estimates(file_counter, state_vals, episode=None, alpha=None):
    fig = plt.figure()
    x = range(1, NSTATES + 1)
    plt.plot(x, ACTUAL_STATE_VALUES, label='Optimal')
    if EX62_T_VALS:
        for i, v in enumerate(EX62_T_VALS):
            plt.plot(x, state_vals[i], label=v)  # [s.v for s in states][1:NSTATES + 1]
    else:
        plt.plot(x, state_vals, label='Estimated')  # [s.v for s in states][1:NSTATES + 1]
    plt.xticks(x, ['A', 'B', 'C', 'D', 'E'])
    plt.title('State value estimates')
    plt.xlim((1, NSTATES))
    plt.ylim((0, 1))
    if episode is not None:
        plt.text(4.2, 0.14, 'T = {}'.format(episode))
    if alpha is not None:
        plt.text(4.2, 0.10, u'$\\alpha$ = {}'.format(round(alpha, 4)))
    plt.legend(title='lambda')
    plt.legend(loc=2)
    fig.savefig(os.path.join('output', OUTPUT_DIR, '{}.png'.format(str(file_counter).zfill(4))))
    plt.close()


def fig4_frame(lambda_vals, alpha_vals, errors, i):
    fig = plt.figure()
    for j in range(len(errors)):
        plt.plot(alpha_vals, errors[j], label=lambda_vals[j])
    plt.xlim((-0.05, 0.65))
    plt.ylim((0.05, 0.75))
    plt.xlabel(u'$\\alpha$')
    plt.ylabel('ERROR')
    plt.text(0.4, 0.1, 'episode limit = {}'.format(i))
    fig.savefig(os.path.join('output', OUTPUT_DIR, '{}.png'.format(str((i / 4) - 1).zfill(4))))
    plt.close()
