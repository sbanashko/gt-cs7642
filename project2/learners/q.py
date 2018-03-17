import numpy as np

from learners.rl_agent import RLAgent
from util import *
from util.helper import discretize_state
from util.logging import logger
from util.plotting import plot


class QLearningAgent(RLAgent):
    """Q"""

    def __init__(self, ns, na, alpha=ALPHA, alpha_decay_rate=1.0,
                 epsilon=EPSILON, epsilon_decay_rate=EPSILON_DECAY_RATE,
                 gamma=GAMMA, dyna=0):

        super().__init__()
        self.s = [0. for _ in range(ns)]
        self.a = 0
        self.ns = ns
        self.na = na
        self.Q = np.zeros((ns, na))
        self.alpha = alpha
        self.alpha_decay_rate = alpha_decay_rate
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma
        self.memory = []
        self.dyna = dyna

        self.random_actions = 0
        self.total_actions = 0

        # Prevent updating Q table
        self.training_mode = True

    def train(self, env):

        # Track rewards
        all_rewards = []

        try:
            while True:
                state = env.reset()
                state = [round(s, PRECISION) for s in state[:CONTINUOUS_OBSERVATIONS]]
                action = self._query_initial(state, env.discrete_obs_space)  # set the state and get first action
                episode_return = 0
                steps = 0
                total_Q_update = 0

                while True:
                    new_state, reward, done, details = env.step(action)
                    new_state = [round(s, PRECISION) for s in new_state[:CONTINUOUS_OBSERVATIONS]]
                    # env.render()
                    episode_return += reward

                    # if steps % 10 == 0:
                    #     print([x for x in new_state])
                    #     print("step {} total_reward {:+0.2f}".format(steps, episode_return))
                    steps += 1

                    if done:
                        break

                    action, delta_Q = self._query(state, action, new_state, reward, env.discrete_obs_space)
                    total_Q_update += delta_Q

                all_rewards.append(episode_return)

                sma = np.mean(all_rewards[-SMA_WINDOW:])

                if self.episodes % 10 == 0:
                    if self.episodes >= SMA_WINDOW:
                        logger.info('Episode {} | Reward = {} | SMA = {}'.format(self.episodes, episode_return, sma))
                    else:
                        logger.info('Episode {} | Reward = {}'.format(self.episodes, episode_return))

                # Convergence
                if self.episodes > SMA_WINDOW and sma >= SOLUTION_THRESHOLD:
                    break

                self.episodes += 1
        except KeyboardInterrupt:
            logger.warn('KeyboardInterrupt - halting training')

        plot(all_rewards, title='Rewards per episode', xlab='Episode', ylab='Reward')
        logger.info('{}% of actions were random'.format(round(100. * self.random_actions / self.total_actions, 2)))

    def _query_initial(self, s, space):
        """
        Select action without updating the Q-table
        :param s:
        :return:
        """
        self.total_actions += 1
        if np.random.random() < self.epsilon:
            self.random_actions += 1
            action = np.random.choice(self.na)
        else:
            action = np.argmax([self.Q[discretize_state(s, space), a] for a in range(self.na)])

        self.v *= self.epsilon_decay_rate

        # Update current state and action
        self.s = s
        self.a = action

        return action

    def _query(self, s, a, sp, r, space):
        """
        Select action and update Q-table
        :param s: previous state
        :param a: selected action
        :param sp: new state
        :param r: immediate reward
        :return:
        """
        delta_Q = self._update_Q((s, a, sp, r), space)

        self.total_actions += 1

        # Dyna-Q
        if self.dyna > 0:

            # Replace T and R models with in-memory historical data
            self.memory.append((self.s, self.a, sp, r))

            # Hallucinate
            for d in range(self.dyna):
                # Update Q-table
                self._update_Q(self.memory[np.random.choice(len(self.memory))], space)

        if np.random.random() < self.epsilon:
            self.random_actions += 1
            action = np.random.choice(self.na)
            self.epsilon *= self.epsilon_decay_rate
        else:
            action = np.argmax([self.Q[discretize_state(sp, space), a] for a in range(self.na)])

        # Update current state and action
        self.s = sp
        self.a = action

        return action, delta_Q

    def _update_Q(self, experience_tuple, space):
        """
        Update Q table
        :param experience_tuple: s, a, s', r
        :return:
        """
        if not self.training_mode:
            return 0
        s, a, sp, r = experience_tuple
        prev_Q = self.Q[discretize_state(s, space), a]
        updated_Q = prev_Q + self.alpha * (
                r + self.gamma * self.Q[discretize_state(sp, space), np.argmax([self.Q[discretize_state(sp, space), i] for i in range(self.na)])] - prev_Q)
        self.Q[discretize_state(s, space), a] = updated_Q
        self.alpha *= self.alpha_decay_rate
        return abs(updated_Q - prev_Q)
