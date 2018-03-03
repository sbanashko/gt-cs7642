from abc import ABC, abstractmethod


class RLAgent(ABC):

    @abstractmethod
    def query_initial(self, *args):
        pass

    @abstractmethod
    def query(self, *args):
        pass
