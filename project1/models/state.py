class State:
    def __init__(self, name, index, v=0.0, r=0.0, e=0.0, terminal=False):
        self.name = name
        self.index = index
        self.v = v
        self.r = r
        self.e = e
        self.terminal = terminal
