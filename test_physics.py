from src.rocket import Rocket
from src.render import Visualization
import numpy as np
from src.dql import DQL
import pickle

# state = pickle.load(open('runs/state.pkl', 'rb'))
test = DQL()
test.test()
breakpoint()
W, H = 600, 760

rocket = Rocket(W, H, prop=0.5)
rocket.set_state(-20, 700, 0.1, 10, -100, 0)

render = Visualization(W, H)
for i in range(5000):
    transition = rocket.update(*(0, 0.7))
    reward = transition[2]
    render.frame(rocket, reward)
    if transition[-1] is True:
        break




