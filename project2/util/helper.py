from util import NBINS


def get_discrete_nstates(env, nbins=NBINS):
    """
    Just for CartPole for now...
    :param env: OpenAI environment
    :param nbins: number of bins (must be same size for all features)
    :return:
    """
    nfeatures = env.observation_space.shape[0]
    return pow(nbins, nfeatures)


def discretize_space(space, nbins=NBINS):
    for x in range(space.shape[0]):
        # Arbitrary min/max that isn't a buhjillion
        space.low[x] = max(-1000, space.low[x])
        space.high[x] = min(1000, space.high[x])
    return space


def discretize_state(s, space, nbins=NBINS):
    """
    Just for CartPole for now...
    :param s: state
    :param space: env observation space
    :param nbins: number of bins (must be same size for all features)
    :return:
    """
    norm_state = [(min(s[x] - space.low[x], 0)) / (space.high[x] - space.low[x]) for x in range(space.shape[0])]
    return int(sum([norm_state[i] * pow(nbins, i) for i in range(len(s))]))
