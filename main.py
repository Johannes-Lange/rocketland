from render import Visualization
from rocket import *

rocket = Rocket(prop=0.5)
# rocket.set_state(0, 700, 0, 0, -100, 0)
rocket.set_state(0, 0, 0, 0, 0, 0)

render = Visualization(600, 760)
for i in range(5000):
    rocket.update(0., 1)
    render.frame(rocket)
    if rocket.terminated():
        break



