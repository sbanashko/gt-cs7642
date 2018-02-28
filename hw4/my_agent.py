import argparse

import gym
import numpy as np
from gym import wrappers, logger
import matplotlib.pyplot as plt


class QLearningAgent(object):
    """Q"""

    def __init__(self, ns, na, alpha=0.2, alpha_decay_rate=0.99, random_action_rate=0.5, random_action_rate_decay=0.99,
                 gamma=0.9):
        self.s = 0
        self.a = 0
        self.ns = ns
        self.na = na
        # Initialize Q table (because it's a finite problem and we can)
        self.Q = np.random.random((ns, na))
        self.alpha = alpha
        self.alpha_decay_rate = alpha_decay_rate
        self.random_action_rate = random_action_rate
        self.random_action_rate_decay = random_action_rate_decay
        self.gamma = gamma

    def act(self, s, a, sp, r):
        """
        Select action and update Q-table
        :param s: previous state
        :param a: selected action
        :param sp: new state
        :param r: immediate reward
        :return:
        """
        self.update_Q((s, a, sp, r))

        # # TODO implement Dyna-Q
        # if self.dyna > 0:
        #
        #     # Replace T and R models with in-memory historical data
        #     self.memory.append((self.s, self.a, sp, r))
        #
        #     # Hallucinate
        #     for d in range(self.dyna):
        #
        #         # Update Q-table
        #         self.Q(self.memory[np.random.choice(len(self.memory))])

        if np.random.random() < self.random_action_rate:
            action = np.random.choice(self.na)
        else:
            action = np.argmax([self.Q[sp, a] for a in range(self.na)])

        self.random_action_rate *= self.random_action_rate_decay

        # Update current state and action
        self.s = sp
        self.a = action

        return action

    def query_initial(self, s):
        """
        Select action without updating the Q-table
        :param s:
        :return:
        """
        if np.random.random() < self.alpha:
            self.alpha *= self.alpha_decay_rate
            return np.random.choice(self.na)
        return max(range(self.na), key=lambda x: self.Q[s, x])

    def update_Q(self, experience_tuple):
        """
        Update Q table
        :param experience_tuple: s, a, s', r
        :return:
        """
        s, a, sp, r = experience_tuple
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (
                r + self.gamma * self.Q[sp, np.argmax([self.Q[sp, i] for i in range(self.na)])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='Taxi-v2', help='Select the environment to run')
    args = parser.parse_args()

    save_results = False  # TODO put this in the argparser

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

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

    all_rewards = []
    done = False

    for e in range(episode_count):

        total_reward = 0
        episode_dir = 'episode_{}'.format(str(e).zfill(4))

        state = env.reset()
        action = agent.query_initial(state)  # set the state and get first action

        for i in range(max_iterations):
            # Execute step
            new_state, reward, done, details = env.step(action)

            # Select next action
            action = agent.act(state, action, new_state, reward)

            if reward > 0:
                print('Finished episode after {}'.format(i))
                env.render()

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

            if done:
                print('Done after {} iterations'.format(i + 1))
                break

            agent.update_Q((state, action, new_state, reward))
            total_reward += reward
            agent.s = new_state

        all_rewards.append(total_reward)

    print(all_rewards)
    plt.plot(all_rewards)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()
