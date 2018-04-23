import numpy as np

from project3.agents import QLearner, FriendQLearner, FoeQLearner, RandomAgent, CEQLearner
from project3.external.emoudahi_env import SoccerEnv
from project3.utils.file_util import save_results
from project3.utils.log_util import logger
from project3.utils.plot_util import plot_results
from project3.vars import *

# Q seed = 0
# Friend-Q seed = 0
# Foe-Q seed =
# uCE-Q seed =
np.random.seed(0)

player = CEQLearner(PLAYER_INFO, NUM_STATES, NUM_ACTIONS)
opponent = RandomAgent(OPPONENT_INFO, NUM_STATES, NUM_ACTIONS)

env = SoccerEnv()

CONTROL_STATE = env.encode_state(0, 2, 0, 1, 0)

control_state_Q_updates = []
actual_updates = 0

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
last_game = False

try:
    while t < MAX_STEPS:

        state = env.reset()
        # Force restart in control state
        env.s = CONTROL_STATE

        if t > MAX_STEPS - 1000 and not last_game:
            logger.info('t = {} : Rendering a late game (hopefully converged by now)'.format(t))
            last_game = True

        states_visited.add(state)
        done = False

        action = player.query_initial(state)
        op_action = opponent.query_initial(state)

        if DEBUG:
            env.render()

        while not done and t < MAX_STEPS:
            t += 1

            if last_game:
                env.render()

            if t % 10000 == 0 and total_games > 0:
                logger.info('{}\t{}\t{}'.format(
                    t,
                    round(1. * player_wins / total_games, 2),
                    round(1. * op_wins / total_games, 2)))

            # 1. Simulate action a_i from state s
            new_state, reward, done, details = env.step(env.encode_action(action, op_action))
            op_reward = -reward

            # 2. Observe action profile a_{-i}, reward R(s,a), and next state s'
            # Done: op_action, op_reward, new_state

            if DEBUG:
                env.render()

            # Quit loop and reset environment
            if done:

                # Manually set terminal state Q value as normalized reward and nothing else
                if isinstance(player, QLearner):
                    player.Q[state, action] = reward / 100.
                else:
                    player.Q[state, action, op_action] = reward / 100.
                if isinstance(opponent, QLearner):
                    opponent.Q[state, op_action] = reward / 100.
                else:
                    opponent.Q[state, op_action, action] = reward / 100.

                delta_Q = 0
                op_delta_Q = 0

                # Track game and win count
                if reward > 0:
                    player_wins += 1
                elif reward < 0:
                    op_wins += 1
                total_games += 1

            # 3. Select policy pi(s') for new state according to selection mechanism f(Q(s'))
            # 4. Update V(s') and Q(s, a) for both players according to selected joint policy
            else:
                action, delta_Q = player.query(state, action, op_action, new_state, reward, opponent.Q)
                op_action, op_delta_Q = opponent.query(state, op_action, action, new_state, op_reward, opponent.Q)

            # Track updates per timestep
            # See Greenwald (2003)
            if state == CONTROL_STATE and action == env.Action.S and (
                    isinstance(player, QLearner) or op_action == env.Action.Stick):
                control_dQ = delta_Q
                actual_updates += 1
            elif len(control_state_Q_updates) > 0:
                control_dQ = control_state_Q_updates[-1]
            else:
                control_dQ = 0

            # 5. Update state s -> s', action a -> a' (DONE: action selected same time as delta_Q)
            state = new_state
            states_visited.add(new_state)

            # 6. Decay alpha (DONE: update inside agent.query())

            control_state_Q_updates.append(control_dQ)
            all_Q_updates.append(delta_Q)
            all_rewards.append(reward)
            all_states_visited.append(len(states_visited))
            all_alphas.append(player.alpha)
            all_rar.append(player.epsilon)

            if done:
                if last_game:
                    last_game = False
                break

except KeyboardInterrupt:
    logger.warn('Gameplay halted after {} timesteps'.format(t))

logger.warn('Actual updates in state s with action SOUTH and op_action STICK: {}'.format(actual_updates))
plot_results(control_state_Q_updates, title=player.algo_name)

save_results(player, control_state_Q_updates)
