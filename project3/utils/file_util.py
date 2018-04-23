import os
from datetime import datetime as dt

import numpy as np


def save_results(agent, arr):
    # Save array of control state Q updates and accompanying agent metadata info
    filename = '{}_{}'.format(dt.now().strftime('%H%M%S'), agent.algo_name)

    # Save
    np.save(os.path.join('.', 'saved_agents', filename), arr)
    f = open(os.path.join('.', 'saved_agents', '{}.txt'.format(filename)), 'w')
    for attr, value in agent.__dict__.items():
        f.write('{}\t{}\n'.format(attr, value))
    f.close()

    return filename
