import os
from datetime import datetime


# Debug
DEBUG = False

# Plot output directory
OUTPUT_DIR = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create output directory
os.mkdir(os.path.join('output', OUTPUT_DIR))

# Number of training sets to use
NSETS = 100

# Number of sequences to run per training set
NEPISODES = 10

# Prevent long-running or infinite loop
MAX_ITERATIONS = 100

# Number of non-terminal states, between 1 and 26 inclusive
NSTATES = 5

# Actual state values, or "ideal predictions"
ACTUAL_STATE_VALUES = [1. * (v + 1) / (NSTATES + 1) for v in range(NSTATES)]

# Weight update locations
# timestep OR episode OR trainset
WEIGHT_UPDATE_LOC = 'trainset'
