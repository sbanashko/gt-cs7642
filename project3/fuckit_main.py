from project3.fuckit import SoccerEnv

from project3.vars import MAX_STEPS

env = SoccerEnv()

# You provide the directory to write to (can be an existing
# directory, including one with existing data -- all monitor files
# will be namespaced). You can also dump to a tempdir if you'd
# like: tempfile.mkdtemp().
# env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)

for e in range(MAX_STEPS):

    total_Q_update = 0
    total_reward = 0

    state = env.reset()
    action = agent.query_initial(state)  # set the state and get first action

    i = 0

    while True:

        # Execute step
        new_state, reward, done, details = env.step(action)

        total_reward += reward

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
