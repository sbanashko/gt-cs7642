import string

from project1.models.state import State
from project1.settings import NSTATES

states = [State(string.ascii_uppercase[i], i + 1, v=0.5, r=0.0) for i in range(NSTATES)]
states.insert(0, State('0', 0, v=0.0, r=0.0, terminal=True))
states.append(State('1', NSTATES + 1, v=0.0, r=1.0, terminal=True))
