from itertools import product
import numpy as np

W, H = 600, 760

POWER_LEVELS = [0., .5, 1.]
ANGLE_LEVELS = [-np.pi/6, 0, np.pi/6]  # pi/18
ACTION = list(product(ANGLE_LEVELS, POWER_LEVELS))

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200  # 400
TARGET_UPDATE = 10

MEMORY_SZ = 10000
MINIBATCH = 50
LR = 1e-5

# (x, y, r, vx, vy, w, x_prop)
# STATE_MINMAX = [(-W/2, W/2), (0, H), (-pi/2, pi/2), (-40, 40), (-110, 50), (-np.pi/6, np.pi/6), (0, 50)]
# STATE_MINMAX = [(-W/2, W/2), (0, H), (0, 1), (-40, 40), (-110, 50), (0, 1), (0, 50)]
STATE_MINMAX = [(0, W/2), (0, H), (0, 1), (0, W/2), (0, H), (0, 1), (0, 50)]
