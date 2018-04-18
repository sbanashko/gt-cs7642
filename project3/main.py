from project3.agents import QLearner
from project3.environment import World
from project3.utils.plot_util import plot_results
from project3.vars import *

player = QLearner(PLAYER_INFO, NUM_STATES, NUM_ACTIONS)
opponent = QLearner(OPPONENT_INFO, NUM_STATES, NUM_ACTIONS)
env = World(player, opponent)

all_Q_updates = []
all_rewards = []
all_states_visited = []

states_visited = set()

i = 0

try:
    while i < MAX_STEPS:

        state = env.reset()
        states_visited.add(state)
        done = False

        # Track game iterations anyway to force reset after boring game
        gamestep = 0

        action = player.query_initial(state)
        op_action = opponent.query_initial(state)

        # print('~~~~~~~~~~ Game Reset ~~~~~~~~~~\n')
        if env.debug:
            env.render()

        while not done and i < MAX_STEPS and gamestep < 500:
            i += 1
            gamestep += 1

            if i % 100000 == 0:
                print(i)

            # Execute step
            new_state, reward, done, details = env.step(action, op_action)
            if env.debug:
                env.render()

            # Quit loop and reset environment
            if done:

                # Manually set terminal state Q value as immediate reward and nothing else
                player.Q[player.s, player.a] = reward

                # Track updates per timestep
                all_Q_updates.append(0)
                all_rewards.append(reward)
                all_states_visited.append(len(states_visited))

                break

            # Select next action
            else:
                action, delta_Q = player.query(state, action, new_state, reward)
                op_action, op_delta_Q = opponent.query(state, op_action, new_state, reward)

                # print('{}\t{}\t{}\t{}'.format(state, action, new_state, reward))
                state = new_state
                states_visited.add(new_state)

                # Track updates per timestep
                all_Q_updates.append(delta_Q)
                all_rewards.append(reward)
                all_states_visited.append(len(states_visited))

except KeyboardInterrupt:
    print('Gameplay halted')

plot_results(all_Q_updates, title='Q-Learner')

# Objective function (uCEQ)
# def utilitarian(N, A, pi_s):
#     [sum(Q[j][s, a]) for j in N]
#     pass

# print(max([2, -1, -5], key=lambda i: pow(i, 2)))

# Rationality constraints
