import os
import random
from collections import deque

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, clone_model
from keras.optimizers import Adam

from util import *
from util.logging import logger
from util.plotting import plot


class DQNAgent:
    def __init__(self, env,
                 alpha=ALPHA,
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 epsilon_decay_rate=EPSILON_DECAY_RATE,
                 epsilon_min=EPSILON_MIN,
                 memory_limit=MEMORY_LIMIT,
                 min_memory_size=MIN_MEMORY_SIZE,
                 net_replacement_freq=NET_REPLACEMENT_FREQ,
                 batch_size=BATCH_SIZE,
                 max_episodes=MAX_EPISODES,
                 use_weights=USE_WEIGHTS):
        self.env = env
        self.n_features = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.learning_rate = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.epsilon_special = epsilon < 0
        self.mem_limit = memory_limit
        self.memory = deque(maxlen=self.mem_limit)
        self.min_memory_size = min_memory_size
        self.net_replacement_freq = net_replacement_freq
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.use_weights = use_weights
        self.weights = 'lander_weights.h5'  # 'hp_lander_{}.h5'.format(epsilon_decay_rate)
        self.Q = self._build_model()
        self.Q_ = clone_model(self.Q)
        self.Q_.set_weights(self.Q.get_weights())
        self.episode_count = 0
        self.solved = False

    def train(self, episodes=-1):

        # Hacky...
        if episodes < 0:
            episodes = self.max_episodes

        episode = 0
        all_rewards = []

        try:
            # Set this to "while True" for genuine convergence
            for e in range(episodes):

                # Start episode
                episode_reward = 0
                self.episode_count = e
                t = 0
                state = self.env.reset()
                state = np.reshape(state, [1, self.n_features])

                while True:
                    # self.env.render()

                    # Select action
                    action = self._select_action(state)

                    # Execute transition
                    next_state, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    next_state = np.reshape(next_state, [1, self.n_features])

                    # Store experience tuple in memory
                    self.memory.append((state, action, reward, next_state, done))
                    state = next_state

                    # Replay using mini batch
                    self._update_Q()

                    # Copy learned Q function into target network
                    if t % self.net_replacement_freq == 0:
                        self.Q_ = clone_model(self.Q)
                        self.Q_.set_weights(self.Q.get_weights())

                    t += 1
                    if done:
                        break

                all_rewards.append(episode_reward)
                sma = np.mean(all_rewards[-SMA_WINDOW:])
                logger.info('{},{},{},{}'.format(episode, episode_reward, self.epsilon, sma))
                episode += 1

                # Uncomment for episodic epsilon decay
                if not self.epsilon_special:
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay_rate

                # Special case: stepwise epsilon decay
                else:
                    if episode < 150:
                        self.epsilon = 1.0
                    elif episode < 250:
                        self.epsilon = 0.5
                    else:
                        self.epsilon = 0.0

                # Convergence
                if sma >= 200:
                    self.solved = True
                    break

        except KeyboardInterrupt:
            logger.info('KeyboardInterrupt: halting training')
        finally:
            plot(all_rewards)
            self._save_model()
            return all_rewards

    def test(self):
        self.epsilon = 0
        return self.train(100)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.n_features, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if self.use_weights and os.path.isfile(self.weights):
            model.load_weights(self.weights)
            self.exploration_rate = 1.0
        return model

    def _save_model(self):
        if self.use_weights:
            self.Q.save(self.weights)

    def _select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        return np.argmax(self.Q.predict(state)[0])

    def _update_Q(self):
        if len(self.memory) < self.min_memory_size:
            return
        sample_batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in sample_batch:
            y = reward
            if not done:
                # Set y according to target network
                y += self.gamma * np.amax(self.Q_.predict(next_state)[0])
            target_val = self.Q.predict(state)
            target_val[0][action] = y
            self.Q.fit(state, target_val, epochs=1, verbose=0)
