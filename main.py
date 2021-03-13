from render import Visualization
from rocket import *
from random import uniform as uf

W, H = 600, 760

rocket = Rocket(W, H, prop=0.5)
# rocket.set_state(0, 700, 0, 0, -100, 0)
rocket.set_state(0, 500, 0, 0, 0, 0)

render = Visualization(W, H)
for i in range(5000):
    ret = rocket.update(*(-0.2, 1))
    render.frame(rocket, ret[2])
    if ret[-1] is True:
        break



