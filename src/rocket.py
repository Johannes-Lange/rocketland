from numpy.linalg import norm
from math import fmod
import numpy as np
import sympy as sp
import random


class Rocket:
    def __init__(self, w, h, prop=1):
        # area
        self.w, self.h = w, h

        # Timestep
        self.t_step = 0.01
        self. lifespan = 0
        self.score = []
        self.previous_shaping = None
        self.dead = False
        self.success = False

        # geometry
        self.length = 41
        self.radius = 1.83
        self.cross_sec = np.pi * self.radius ** 2

        # densitys
        rho_al = 2700  # density aluminum
        self.rho_body = (2 * np.pi * self.radius) * 0.03 * rho_al / self.cross_sec  # 3cm wall thicknes
        self.rho_prop = 700

        # mass, complete rocket is tank
        self.prop = prop
        self.mass, self.center, self.moment_inertia, self.y_low, self.y_top = None, None, None, None, None
        self.x_prop = prop*self.length  # height of propellant
        self.center_gravity()
        self.moment_of_inertia()

        # position and thruster
        self.f_0 = 9 * 540e3  # maximal Thrust in Newton, 9 Merlin C1 (before: 9*420e3)
        self.prop_consumption = (self.length*self.cross_sec)/(self.f_0*170/self.t_step) * 10  # m^3/N, 170s burntime
        self.position = [np.array([0, 200, 0])]  # [np.zeros(3)]  # coordinates + rotation (x, y, R)
        self.thruster = np.zeros(2)  # angle + force (phi, f) (local coordinates)
        self.thruster_level = 0

        self.omega = np.array([0])
        self.v = np.array([[random.randint(-10, 10)],
                           [random.randint(-60, -40)]])
        self.deq = symbolic_equation()  # define differential equation self.deq for movement

    def update(self, angle, power):
        # get state
        state_0 = self.get_state()
        self.lifespan += 1

        # upadte thruster
        self.thruster_level = power
        self.update_thrust(angle, power)

        # update propellant and mass
        self.x_prop -= self.prop_consumption * self.thruster[1] / self.cross_sec

        # update center + moment inertia
        self.center_gravity()
        self.moment_of_inertia()

        # update position
        self.update_position()

        # lowest point (check if termination)
        self.lowest_point()
        self.terminated()

        self.score.append(self.update_score())

        # state_0, action, reward, state_1, terminal
        state_1 = self.get_state()
        return state_0, (angle, power), self.score[-1], state_1, self.dead

    def update_score(self):
        # depends on velocity, rotation, distance to 0
        state = self.get_state()
        reward = np.array([0])

        shaping = np.array([- 100 * norm(state[0:2]),  # distance
                            - 100 * norm(state[3:5]),  # velocity (200, maybe lower scaling)
                            - 100 * norm(state[2])])  # rotation

        if self.previous_shaping is not None:
            reward = shaping - self.previous_shaping
        self.previous_shaping = shaping
        reward = reward.sum()
        reward -= 0.1 * abs(state[2])

        if self.success:
            reward += 200

        return reward

    def terminated(self):
        epsilon = np.arcsin(self.radius / self.center)
        x, y, r = self.position[-1][0], self.position[-1][1], self.position[-1][2]
        out_area = False
        ground = False
        success = False

        # boarders
        if np.abs(x) > self.w/2 + 10 or np.abs(y) > self.h + 10:
            out_area = True

        # ground
        if self.y_low < 0 or self.y_top < 0:
            ground = True
            if r % (2 * np.pi) < epsilon and norm(self.v < 5):
                # landing successfull if no tip over and slow enough
                success = True

        self.dead = out_area or ground
        self.success = success

    """ *********** Physics *********** """
    def update_thrust(self, angle, p):
        f = p*self.f_0

        self.thruster[0] = angle
        if self.x_prop > 0:
            self.thruster[1] = f
        else:
            self.thruster[1] = 0

    def center_gravity(self):
        m_body = self.rho_body * self.cross_sec * self.length
        m_prop = self.rho_prop * self.cross_sec * self.x_prop
        self.mass = m_body + m_prop
        self.center = (m_body * self.length/2 + m_prop * self.x_prop/2) / (m_body + m_prop)

    def moment_of_inertia(self):
        self.moment_inertia = 1/3*self.cross_sec*(self.rho_body*((self.length-self.center)**3-(-self.center)**3) +
                                                  self.rho_prop*((self.x_prop-self.center)**3-(-self.center)**3))

    def update_position(self):
        v_x_new, v_y_new, w_new = self.deq(self.center, self.moment_inertia, self.omega.item(), self.v[0].item(),
                                           self.v[1].item(), self.t_step, self.thruster[0], self.thruster[1],
                                           self.mass, self.position[-1][2])
        # save solution for velocity
        self.omega = np.array([w_new])
        self.v = np.array([[v_x_new],
                           [v_y_new]])
        # update position
        last_position = self.position[-1]
        self.position.append(last_position + np.array([v_x_new, v_y_new, w_new])*self.t_step)

    def rotation_matrix(self):
        return np.array([[np.cos(self.position[-1][2]), np.sin(self.position[-1][2])],
                         [-np.sin(self.position[-1][2]), np.cos(self.position[-1][2])]])

    def lowest_point(self):
        x, y, r = self.position[-1][0], self.position[-1][1], self.position[-1][2]
        y_b = self.position[-1][1] - np.cos(r) * self.center
        y_t = self.position[-1][1] + np.cos(r) * (self.length - self.center)
        self.y_low = y_b
        self.y_top = y_t

    """ *********** Get / Set *********** """
    def get_state(self):
        state = np.concatenate([self.position[-1][0].ravel()/(self.w/2),  # x
                                self.position[-1][1].ravel()/self.h,  # y
                                self.position[-1][2].ravel(),  # r
                                self.v[0].ravel()/(self.w/2)*self.t_step * 100,  # vx
                                self.v[1].ravel()/self.h*self.t_step * 100,  # vy
                                self.omega.ravel()/self.t_step*20,  # omega
                                np.array([self.x_prop]).ravel()/40])  # c_prop

        return state

    def set_state(self, x, y, r, v_x, v_y, w):
        self.position.append(np.array([x, y, r]))
        self.x_prop = self.prop * self.length
        self.center_gravity()
        self.moment_of_inertia()
        self.v = np.array([[v_x],
                           [v_y]])
        self.omega = np.array([w])
        self.lifespan = 0
        self.score = []
        self.dead = False


def symbolic_equation():
    r, v_x, v_x_l, v_y, v_y_l, t, phi, f, m, j_m, w, w_l, x_c = sp.symbols('r v_x v_x_l v_y v_y_l t \
                                                                           phi f m j_m w w_l x_c')
    # rotation matrix
    rot = sp.Matrix([[sp.cos(r), sp.sin(r)],
                     [-sp.sin(r), sp.cos(r)]])
    # gravity
    f_g = -rot * sp.Matrix([[0],
                           [-9.81 * m]])
    # thrust
    f_t = sp.Matrix([[f * sp.sin(phi)],
                     [-f * sp.cos(phi)]])

    # inertia force
    f_in = m * rot * sp.Matrix([[(v_x - v_x_l) / t],
                                [(v_y - v_y_l) / t]])

    # inertia moment
    m_in = j_m * (w - w_l) / t
    # moment from thrust
    m_t = f_t[0] * x_c

    # Equations to solve
    force_u = f_g[0] + f_in[0] + f_t[0]
    force_v = f_g[1] + f_in[1] + f_t[1]
    moment = m_in + m_t

    eq1 = sp.Eq(force_u, 0)
    eq2 = sp.Eq(force_v, 0)
    eq3 = sp.Eq(moment, 0)

    # solution with variables
    sol, = sp.linsolve((eq1, eq2, eq3), (v_x, v_y, w))
    eq = sp.lambdify([x_c, j_m, w_l, v_x_l, v_y_l, t, phi, f, m, r], sol)
    return eq
