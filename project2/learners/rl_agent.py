from abc import ABC, abstractmethod


class RLAgent(ABC):

    def __init__(self):
        self.solved = False
        self.episodes = 0

    @abstractmethod
    def train(self, *args):
        pass
