from render import Visualization
from rocket import *
from random import uniform as uf

W, H = 600, 760

rocket = Rocket(prop=0.5)
rocket.set_state(0, 700, 0, 0, -100, 0)
# rocket.set_state(0, 15, 0, 0, 0, 0)

render = Visualization(W, H)
for i in range(5000):
    # rocket.update(uf(-0.2, 0.2), uf(0.6, 1))
    ret = rocket.update(0, 0.84)
    render.frame(rocket, ret[3])
    if ret[-1] is True:
        break



