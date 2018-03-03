import os
import sys

import numpy as np


def get_samples():
    return [{
        'lookup': (462, 4),
        'expected': -11.374402515,
    }, {
        'lookup': (398, 3),
        'expected': 4.348907,
    }, {
        'lookup': (253, 0),
        'expected': -0.5856821173,
    }, {
        'lookup': (377, 1),
        'expected': 9.683,
    }, {
        'lookup': (83, 5),
        'expected': -12.8232660372,
    }]


def validate_results(Q):
    problems = get_samples()
    for i, p in enumerate(problems):
        actual = Q[p['lookup']]
        result = '✔' if np.isclose(p['expected'], actual) else 'X'
        print()
        print('{} Problem {}'.format(result, i))
        print('Expected: {}'.format(p['expected']))
        print('Actual: {}'.format(actual))

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
