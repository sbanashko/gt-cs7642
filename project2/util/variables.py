from datetime import datetime


##############################################################################
# Global configs
##############################################################################

# Default environment
DEFAULT_ENV = 'LunarLander-v2'

# Execution timestamp
RUN_TIMESTAMP = datetime.now()

# State discretization precision (decimal places)
PRECISION = 3

# x, y, vx, vy, θ, vθ
CONTINUOUS_OBSERVATIONS = 6

# left leg, right leg
DISCRETE_OBSERVATIONS = 2

# Moving average
SMA_WINDOW = 100

# Solution score threshold for moving average
SOLUTION_THRESHOLD = 200

# Render environment every x frames
RENDER_INTERVAL = 30

# Stop even if not converged
MAX_EPISODES = 1000  # 500

# Load and save weights from/to disk
USE_WEIGHTS = True

##############################################################################
# Hyperparameters
##############################################################################

ALPHA = 0.0001  # 0.0001
GAMMA = 0.9  # 0.9
MEMORY_LIMIT = 100000  # 100000
MIN_MEMORY_SIZE = 5000  # 5000
NET_REPLACEMENT_FREQ = 1000  # 4
BATCH_SIZE = 32  # 32
EPSILON = 1.0  # 1.0
EPSILON_DECAY_RATE = 0.995  # 0.995
EPSILON_MIN = 0.1  # 0.1
EPOCHS = 1
NBINS = 10
