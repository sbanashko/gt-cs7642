#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot(x, y, xlab=u'$\lambda$', ylab='ERROR'):
    """
    Error vs lambda, constant alpha
    :return:
    """
    plt.plot(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def plot_fig4():
    """
    Error vs alpha, one plot per lambda value
    :return:
    """
    pass


def plot_fig5():
    """
    Error vs lambda, best alpha per lambda value
    :return:
    """
    pass


def plot_value_updates():
    pass


def plot_td_mc_comparison():
    pass
