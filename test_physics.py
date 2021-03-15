from src.rocket import Rocket
from src.render import Visualization

W, H = 600, 760

rocket = Rocket(W, H, prop=0.5)
rocket.set_state(0, 700, 0.2, -20, -120, -0.2)

render = Visualization(W, H)
for i in range(5000):
    transition = rocket.update(*(0.05, .8))
    reward = transition[2]
    render.frame(rocket, reward)
    if transition[-1] is True:
        break




