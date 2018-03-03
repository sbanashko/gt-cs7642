import argparse

import gym
from gym import wrappers, logger

from Q_learner import QLearningAgent
from util import *

parser = argparse.ArgumentParser(description=None)
parser.add_argument('env_id', nargs='?', default='Taxi-v2', help='Select the environment to run')
args = parser.parse_args()

# You can set the level to logger.DEBUG or logger.WARN if you
# want to change the amount of output.
logger.set_level(logger.WARN)

env = gym.make(args.env_id)

# You provide the directory to write to (can be an existing
# directory, including one with existing data -- all monitor files
# will be namespaced). You can also dump to a tempdir if you'd
# like: tempfile.mkdtemp().
outdir = 'output/tmp/q-agent-results'
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
agent = QLearningAgent(env.unwrapped.nS, env.unwrapped.nA)

episode_count = 1000
max_iterations = 1000

# Various trackers
all_Q_updates = []
all_rewards = []
all_iters_per_episode = []
all_rars = []

sample1 = []
sample2 = []
sample3 = []
sample4 = []
sample5 = []

for e in range(episode_count):

    total_Q_update = 0
    total_reward = 0
    episode_dir = 'E{}'.format(str(e).zfill(4))

    state = env.reset()
    action = agent.query_initial(state)  # set the state and get first action

    i = 0

    while True:

        all_rars.append(agent.random_action_rate)

        # Execute step
        new_state, reward, done, details = env.step(action)

        # TODO subroutine updates (see Diettrich Fig 2, p238)
        # https://www.jair.org/media/639/live-639-1834-jair.pdf
        # taxirow, taxicol, passidx, destidx = env.unwrapped.decode(new_state)

        total_reward += reward

        # Update samples for graphing/debugging
        sample1.append(agent.Q[462, 4])
        sample2.append(agent.Q[398, 3])
        sample3.append(agent.Q[253, 0])
        sample4.append(agent.Q[377, 1])
        sample5.append(agent.Q[83, 5])

        # Quit loop and reset environment
        if done or reward == 20:
            # Manually set terminal state Q value as immediate reward and nothing else
            agent.Q[agent.s, agent.a] = reward
            break

        # Select next action
        else:
            action, delta_Q = agent.query(state, action, new_state, reward)
            # Add Q update value to tracker
            total_Q_update += delta_Q
            i += 1

    logger.warn('Episode {}: {} iterations'.format(e + 1, i + 1))
    all_Q_updates.append(total_Q_update)
    all_rewards.append(total_reward)
    all_iters_per_episode.append(i)

plot_results(all_Q_updates, all_rewards)

# Validate Q values against HW4 sheet
validate_results(agent.Q)

# Print Q values for HW4 problems
show_hw_answers(agent.Q)

# PLOT ALL THE THINGS
samples = get_samples()
sample_range = range(len(sample1))

plt.plot(sample1, 'r-')
plt.plot([samples[0]['expected'] for _ in sample_range], 'r--')
plt.plot(sample2, 'm-')
plt.plot([samples[1]['expected'] for _ in sample_range], 'm--')
plt.plot(sample3, 'y-')
plt.plot([samples[2]['expected'] for _ in sample_range], 'y--')
plt.plot(sample4, 'g-')
plt.plot([samples[3]['expected'] for _ in sample_range], 'g--')
plt.plot(sample5, 'b-')
plt.plot([samples[4]['expected'] for _ in sample_range], 'b--')
plt.title('Sample Q Values')
plt.show()

plt.plot(all_rars)
plt.title('Random action rates')
plt.show()

plt.plot(all_iters_per_episode)
plt.title('Iterations per episode')
plt.show()

# Close the env and write monitor result info to disk
env.close()
