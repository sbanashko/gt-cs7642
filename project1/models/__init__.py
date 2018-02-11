import string

from project1.models.state import State
from project1.settings import NUM_STATES

states = [State(string.ascii_uppercase[i], v=0.5, r=0.0) for i in range(NUM_STATES)]
states.insert(0, State('0', v=0.0, r=0.0, terminal=True))
states.append(State('1', v=0.0, r=1.0, terminal=True))
