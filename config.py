from itertools import product
import numpy as np

# Width, height of game area
W, H = 600, 760

# Possible action combinations
POWER_LEVELS = [0., .5, 1.]
ANGLE_LEVELS = [-np.pi/6, 0, np.pi/6]
ACTION = list(product(ANGLE_LEVELS, POWER_LEVELS))

# EPS = EPS_END + (EPS_START - EPS_END) * exp(- game / EPS_DECAY)
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Update target nets parameters every TARGET games to policy nets parameters
TARGET_UPDATE = 10

# Keep last MEMORY_SZ transitions to sample minibatches
MEMORY_SZ = 30000
MINIBATCH = 50
LR = 1e-5
