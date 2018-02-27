import argparse
import os
import sys
from datetime import datetime

import gym
import numpy as np
from gym import wrappers, logger

from util import save_txt


class QLearningAgent(object):
    """Q"""

    def __init__(self, ns, na, alpha=0.2, alpha_decay_rate=0.99, gamma=0.9):
        self.ns = ns
        self.na = na
        # Initialize Q table (because it's a finite problem and we can)
        self.Q = np.random.random((ns, na))
        self.alpha = alpha
        self.alpha_decay_rate = alpha_decay_rate
        self.gamma = gamma

    def act(self, s, a, r):
        # Now choose next action
        if np.random.random() < self.alpha:
            self.alpha *= self.alpha_decay_rate
            return np.random.choice(self.na)
        return max(range(self.na), key=lambda x: self.Q[s, x])

    def update_q(self, s, a, r, sp):
        # Update Q table
        q_estimate = r + self.gamma * self.Q[sp, max(range(self.na), key=lambda x: self.Q[s, x])]
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * q_estimate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Taxi-v2', help='Select the environment to run')
    args = parser.parse_args()

    save_results = False  # TODO put this in the argparser

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    if save_results:
        # Make new output directory
        OUTPUT_DIR = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.mkdir(os.path.join('output', OUTPUT_DIR))

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'output/tmp/q-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = QLearningAgent(env.unwrapped.nS, env.unwrapped.nA)

    episode_count = 100
    max_iterations = 1000

    action = 0
    reward = 0
    done = False

    for e in range(episode_count):
        episode_dir = 'episode_{}'.format(str(e).zfill(4))

        if save_results:
            os.mkdir(os.path.join('output', OUTPUT_DIR, episode_dir))

        state = env.reset()
        for i in range(max_iterations):
            action = agent.act(state, action, reward)
            new_state, reward, done, details = env.step(action)
            agent.update_q(state, action, reward, new_state)

            if save_results:
                save_txt(env, OUTPUT_DIR, episode_dir, i)

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

            if done:
                print('Done after {} iterations'.format(i + 1))
                break

    # Close the env and write monitor result info to disk
    env.close()
