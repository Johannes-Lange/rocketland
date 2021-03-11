import matplotlib.pyplot as plt
from rocket import *
import random

# first stage, dry and propellant mass right
rho_prop = 700
l_r = 41
r = 1.83

rocket = Rocket(rho_prop, l_r, r)
# print(rocket.x_prop)
# print(rocket.mass)

height = []
mass = []

for _ in range(300):
    rocket.update(0, 0)
    print(rocket.position[-1])
    height.append(rocket.position[-1][1])
    mass.append(rocket.mass)

plt.figure()
plt.plot(height)
plt.show()


