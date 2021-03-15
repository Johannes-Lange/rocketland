from itertools import product
import numpy as np

W, H = 600, 760

POWER_LEVELS = [0., .5, .6, .7, .8, 1.]
ANGLE_LEVELS = [-np.pi/6, -np.pi/18, 0, np.pi/18, np.pi/6]
ACTION = list(product(ANGLE_LEVELS, POWER_LEVELS))

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 400

MEMORY_SZ = 10000
MINIBATCH = 40

# (x, y, r, vx, vy, w, x_prop)
# STATE_MINMAX = [(0, 40), (-600 / 2, 600 / 2), (0, 760), (-np.pi/3, np.pi/3), (-40, 40), (-120, 50), (-1.2, 1.2)]
STATE_MINMAX = [(-W/2, W/2), (0, H), (-np.pi/2, np.pi/2), (-40, 40), (-110, 50), (-np.pi/6, np.pi/6), (0, 50)]