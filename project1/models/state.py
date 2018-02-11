class State:
    def __init__(self, name, v=0, r=0, e=0, terminal=False):
        self.name = name
        self.v = v
        self.r = r
        self.e = e
        self.terminal = terminal
