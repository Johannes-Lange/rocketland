from rocket import *

# first stage, dry and propellant mass right
rho_prop = 700
l_r = 41
r = 1.83

rocket = Rocket(rho_prop, l_r, r)
# print(rocket.x_prop)
# print(rocket.mass)

for _ in range(5000):
    rocket.update(0, 1)
    print(rocket.position[-1])
    print(rocket.mass)
# print('Höhe', rocket.x_prop)
#
# print(rocket.mass * 9.81)
# print(rocket.thruster)


