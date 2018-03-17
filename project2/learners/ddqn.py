from learners.rl_agent import RLAgent


class DoubleDQNAgent(RLAgent):
    def __init__(self, replay_memory, replay_limit, theta, batch_size, net_replacement_freq):
        self.replay_memory = replay_memory
        self.replay_limit = replay_limit
        self.theta = theta
        self.theta_0 = theta
        self.batch_size = batch_size
        self.net_replacement_freq = net_replacement_freq

    def train(self, *args):
        pass

    def test(self, *args):
        pass
