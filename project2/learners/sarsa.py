from learners.rl_agent import RLAgent


class SarsaAgent(RLAgent):
    def __init__(self, ns, na):
        self.ns = ns
        self.na = na

    def query_initial(self):
        pass

    def query(self):
        pass
