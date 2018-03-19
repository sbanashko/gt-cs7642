import argparse
import os
from datetime import datetime
import numpy as np

import Box2D
# from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces, wrappers, logger
from gym.utils import seeding

from learners.dqn import DQNAgent
from learners.q import QLearningAgent
from util import *
from util.helper import get_discrete_nstates, discretize_space
from util.plotting import plot

parser = argparse.ArgumentParser(description=None)
parser.add_argument('env_id', nargs='?', default=DEFAULT_ENV, help='Select the environment to run')
parser.add_argument('-l', '--learner', default='dqn', choices=['q', 'dqn'], help='Select agent to train with')

# hyperparameters
parser.add_argument('--alpha', default=ALPHA, type=float, help='NN learning rate')
parser.add_argument('--gamma', default=GAMMA, type=float, help='Discount rate')
parser.add_argument('--epsilon', default=EPSILON, type=float, help='Random action rate')
parser.add_argument('--edr', default=EPSILON_DECAY_RATE, type=float, help='Epsilon decay rate')
parser.add_argument('--minepsilon', default=EPSILON_MIN, type=float, help='Minimum epsilon value')
parser.add_argument('--memlimit', default=MEMORY_LIMIT, type=int, help='DQN buffer size')
parser.add_argument('--nrf', default=NET_REPLACEMENT_FREQ, type=int, help='NN weight replacement frequency')
parser.add_argument('--batchsize', default=BATCH_SIZE, type=int, help='Mini-batch sampling size')
parser.add_argument('--maxepisodes', default=MAX_EPISODES, type=int, help='Mini-batch sampling size')

args = parser.parse_args()

env = gym.make(args.env_id)
# env = wrappers.Monitor(env, directory='output', force=True)

if __name__ == "__main__":

    # for edr in [args.edr]:
    #
    #     print('*' * 80)
    #     print('epsilon = {}'.format(edr))
    #     print('*' * 80)

    agent = None
    weights = None

    if args.learner == 'q':

        # Discretize states first
        ns = get_discrete_nstates(env)

        # Save discrete observation space
        env.discrete_obs_space = discretize_space(env.observation_space)

        # Q-Learning Agent
        agent = QLearningAgent(ns, env.action_space.n)

    elif args.learner == 'dqn':

        # Deep Q Network Agent
        agent = DQNAgent(env,
                         alpha=args.alpha,
                         gamma=args.gamma,
                         epsilon=args.epsilon,
                         epsilon_decay_rate=args.edr,  # args.edr
                         epsilon_min=args.minepsilon,
                         memory_limit=args.memlimit,
                         net_replacement_freq=args.nrf,
                         batch_size=args.batchsize,
                         max_episodes=args.maxepisodes)

        # output_dir = os.path.join('output', datetime.strftime(RUN_TIMESTAMP, '%y%m%d%H%M%S'))
        # os.mkdir(os.path.join('..', output_dir))
        # rewards = []
        # edrs = [0.999, 0.99, 0.9]

        # for edr in edrs:
        #     agent = DQNAgent('LunarLander-v2', epsilon_decay_rate=edr)
        #     rewards.append(agent.train())

        # agent = DQNAgent('LunarLander-v2', epsilon_special=True)
        # plot(rewards, key=edrs, save=True)

    else:
        print('Invalid learner type')

    if agent is not None:
        initial_state = env.reset()

        try:
            agent.train()
        except KeyboardInterrupt:
            logger.info('KeyboardInterrupt - finishing program')

        result_str = 'alpha={} ' \
                     'gamma={} ' \
                     'memory_limit={} ' \
                     'net_replacement_freq={} ' \
                     'batch_size={} ' \
                     'epsilon={} ' \
                     'epsilon_decay_rate={}'.format(
            ALPHA,
            GAMMA,
            MEMORY_LIMIT,
            NET_REPLACEMENT_FREQ,
            BATCH_SIZE,
            EPSILON,
            EPSILON_DECAY_RATE
        )

        with open(os.path.join('output', 'results.txt'), 'a') as output_file:
            output_file.write('{} : {} episodes {}\n'.format(datetime.strftime(RUN_TIMESTAMP, '%y%m%d_%H%M%S'),
                                                             agent.episode_count,
                                                             '(converged)' if agent.solved else ''))
            output_file.write('{}\n'.format(result_str))
            output_file.write('\n')

    else:
        pass
