import numpy as np


# Thanks Miguel!
np.set_printoptions(suppress=True)

# Debug
DEBUG = False

# Plot output directory
OUTPUT_DIR = 'output'

# Number of training sets to use
TRAINING_SETS = 1000

# Number of sequences to run per training set
NUM_EPISODES = 10

# Number of non-terminal states, between 1 and 26 inclusive
NUM_STATES = 5

# Actual state values, or "ideal predictions"
ACTUAL_STATE_VALUES = [1. * v / (NUM_STATES + 1) for v in range(1, NUM_STATES + 1)]
