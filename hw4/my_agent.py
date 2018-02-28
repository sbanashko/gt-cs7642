import argparse

import gym
from gym import wrappers, logger

from Q_learner import QLearningAgent
from util import *


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

episode_count = 1000
max_iterations = 1000

# Various trackers
all_Q_updates = []
all_rewards = []
all_iters_per_episode = []
all_rars = []

done = False

for e in range(episode_count):

    total_Q_update = 0
    total_reward = 0
    episode_dir = 'episode_{}'.format(str(e).zfill(4))

    state = env.reset()
    action = agent.query_initial(state)  # set the state and get first action

    i = 0

    while True:

        i += 1

        all_rars.append(agent.random_action_rate)

        # Execute step
        new_state, reward, done, details = env.step(action)

        # TODO subroutine updates (see Diettrich Fig 2, p238)
        # https://www.jair.org/media/639/live-639-1834-jair.pdf
        taxirow, taxicol, passidx, destidx = env.unwrapped.decode(new_state)

        # Select next action
        action, delta_Q = agent.query(state, action, new_state, reward)

        # Add Q update value to tracker
        total_Q_update += delta_Q

        if e == episode_count - 1:
            env.render()

        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.

        total_reward += reward
        agent.s = new_state

        if done:
            print('Done after {} iterations'.format(i))
            break

    all_Q_updates.append(total_Q_update)
    all_rewards.append(total_reward)
    all_iters_per_episode.append(i)


plot_results(all_Q_updates, all_rewards)

# Validate Q values against HW4 sheet
validate_results(agent.Q)

plt.plot(all_rars)
plt.title('Random action rates')
plt.show()

plt.plot(all_iters_per_episode)
plt.title('Iterations per episode')
plt.show()

# Close the env and write monitor result info to disk
env.close()
