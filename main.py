from render import Visualization
from rocket import *
import numpy as np
from dql import DQL

a = DQL(vis=False)
a.iteration(2)
breakpoint()
W, H = 600, 760

rocket = Rocket(W, H, prop=0.5)
# rocket.set_state(0, 700, 0, 0, -100, 0)
rocket.set_state(0, 700, 0.2, -20, -120, -0.2)

render = Visualization(W, H)
for i in range(5000):
    if i > 100:
        ret = rocket.update(*(0, 1))
    else:
        ret = rocket.update(*(-np.pi/18, .3))
    render.frame(rocket, ret[2])
    if ret[-1] is True:
        break



