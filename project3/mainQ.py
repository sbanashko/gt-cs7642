from project3.agents import QLearner, FriendQLearner, FoeQLearner, RandomAgent
from project3.environment import World
from project3.utils.log_util import logger
from project3.utils.plot_util import plot_results
from project3.vars import *
import numpy as np


np.random.seed(1)

player = FoeQLearner(PLAYER_INFO, NUM_STATES, NUM_ACTIONS)
opponent = RandomAgent(OPPONENT_INFO, NUM_STATES, NUM_ACTIONS)

env = World(player, opponent, debug=DEBUG)

control_state_Q_updates = []
all_Q_updates = []
all_rewards = []
all_states_visited = []
all_alphas = []
all_rar = []

states_visited = set()

t = 0
player_wins = 0
op_wins = 0
total_games = 0

try:
    while t < MAX_STEPS:

        state = env.reset()
        states_visited.add(state)
        done = False

        action = player.query_initial(state)
        op_action = opponent.query_initial(state)

        # print('~~~~~~~~~~ Game Reset ~~~~~~~~~~\n')
        if env.debug:
            env.render()

        while not done and t < MAX_STEPS:
            t += 1

            if t % 1000 == 0 and total_games > 0:
                logger.info('{}\t{}\t{}'.format(
                    t,
                    round(1. * player_wins / total_games, 2),
                    round(1. * op_wins / total_games, 2)))

            # Execute step (zero-sum game)
            new_state, reward, done, details = env.step(action, op_action)
            op_reward = -reward

            if env.debug:
                env.render()

            # Quit loop and reset environment
            if done:

                # Manually set terminal state Q value as immediate reward and nothing else
                if isinstance(player, QLearner):
                    player.Q[state, action] = reward
                else:
                    player.Q[state, action, op_action] = reward
                if isinstance(opponent, QLearner):
                    opponent.Q[state, op_action] = op_reward
                else:
                    opponent.Q[state, op_action, action] = op_reward

                delta_Q = 0
                op_delta_Q = 0

                # Trackers
                if reward > 0:
                    player_wins += 1
                elif reward < 0:
                    op_wins += 1
                total_games += 1

            # Select next action
            else:
                if type(player) == QLearner:
                    action, delta_Q = player.query(state, action, new_state, reward)
                else:
                    action, delta_Q = player.query(state, action, op_action, new_state, reward)

                if type(opponent) == QLearner:
                    op_action, op_delta_Q = opponent.query(state, op_action, new_state, op_reward)
                else:
                    op_action, op_delta_Q = opponent.query(state, op_action, action, new_state, op_reward)

            # Track updates per timestep
            # See Greenwald (2008)
            if state == CONTROL_STATE and action == SOUTH and (isinstance(player, QLearner) or op_action == STICK):
                    control_state_Q_updates.append(delta_Q)
            elif len(control_state_Q_updates) > 0:
                control_state_Q_updates.append(control_state_Q_updates[-1])
            else:
                control_state_Q_updates.append(0)

            state = new_state
            states_visited.add(new_state)

            all_Q_updates.append(delta_Q)
            all_rewards.append(reward)
            all_states_visited.append(len(states_visited))
            all_alphas.append(player.alpha)
            all_rar.append(player.random_action_rate)

            if done:
                break
        # break

except KeyboardInterrupt:
    print('Gameplay halted after {} timesteps'.format(t))

plot_results(control_state_Q_updates, title=player.algo_name)

# plot_results(all_Q_updates, title=player.algo_name)

# Objective function (uCEQ)
# def utilitarian(N, A, pi_s):
#     [sum(Q[j][s, a]) for j in N]
#     pass

# print(max([2, -1, -5], key=lambda i: pow(i, 2)))

# Rationality constraints
