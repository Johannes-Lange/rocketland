from numpy.linalg import norm
import numpy as np
import sympy as sp
import random


class Rocket:
    def __init__(self, prop=1):
        # Timestep
        self.t_step = 0.01

        # score
        self.score = 0
        self.bins = np.array([0, 5, 15, 30, 80, 150, 300, 1000])

        # geometry
        self.length = 41
        self.radius = 1.83
        self.cross_sec = np.pi * self.radius ** 2

        # densitys
        rho_al = 2700  # density aluminum
        self.rho_body = (2 * np.pi * self.radius) * 0.03 * rho_al / self.cross_sec  # 3cm wall thicknes
        self.rho_prop = 700

        # mass, complete rocket is tank
        self.mass, self.center, self.moment_inertia, self.y_low = None, None, None, None
        self.x_prop = prop*self.length  # height of propellant
        self.center_gravity()
        self.moment_of_inertia()

        # position and thruster
        self.f_0 = 9 * 420e3  # maximal Thrust in Newton, 9 Merlin C1
        self.prop_consumption = (self.length*self.cross_sec)/(self.f_0*170/self.t_step)  # m^3/N, 170s burntime
        self.position = [np.array([0, 200, 0])]  # [np.zeros(3)]  # coordinates + rotation (x, y, R)
        self.thruster = np.zeros(2)  # angle + force (phi, f) (local coordinates)

        self.omega = np.array([0])
        self.v = np.array([[random.randint(-10, 10)],
                           [random.randint(-60, -40)]])
        self.deq = symbolic_equation()  # define differential equation self.deq for movement

    def update(self, angle, power):
        # get state
        state_0 = self.get_state()

        # upadte thruster
        self.update_thrust(angle, power)

        # update propellant and mass
        self.x_prop -= self.prop_consumption * self.thruster[1] / self.cross_sec

        # update center + moment inertia
        self.center_gravity()  # updates mass and center of gravity
        self.moment_of_inertia()

        # update position
        self.movement()

        # lowest point (check if termination)
        self.lowest_point()
        self.update_score()

        # state_0, action, reward, state_1, terminal
        state_1 = self.get_state()
        return state_0, angle, power, self.update_score(), state_1, self.terminated()

    def movement(self):
        v_x_new, v_y_new, w_new = self.deq(self.center, self.moment_inertia, self.omega.item(), self.v[0].item(),
                                           self.v[1].item(), self.t_step, self.thruster[0], self.thruster[1],
                                           self.mass, self.position[-1][2])
        self.omega = np.array([w_new])
        self.v = np.array([[v_x_new],
                           [v_y_new]])
        last_position = self.position[-1]
        self.position.append(last_position + np.array([v_x_new, v_y_new, w_new])*self.t_step)

    def update_thrust(self, angle, p):
        # throttle down to 20% possible --> p between 0.2 and 1
        # F = mass_flow * velocity (+ (p_exit - p_0) * A_exit)
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

    def rotation_matrix(self):
        return np.array([[np.cos(self.position[-1][2]), np.sin(self.position[-1][2])],
                         [-np.sin(self.position[-1][2]), np.cos(self.position[-1][2])]])

    def update_score(self):
        # depends on velocity, rotation, distance to 0, leverage of velocity increases with lower distance
        vx, vy, w = self.v[0], self.v[1], self.omega
        x, y, r = self.position[-1][0], self.position[-1][1], self.position[-1][2]

        sc = 0
        if -100 < vy < 0:
            sc += 1
        if np.abs(vx) < 5:
            sc += 1
        if r < 0.2:
            idx = np.digitize(y, self.bins).item()
            sc += {0: 10, 1: 9, 2: 8, 3: 7, 4: 5, 5: 2, 6: 1, 7: 0}[idx]
            if sc > 8:
                if -40 < vy < 0:
                    sc += (5 - int(np.abs(vy)/10))
        if vy > 1:
            sc = -1
        print(sc)
        return sc

    def lowest_point(self):
        x, y, r = self.position[-1][0], self.position[-1][1], self.position[-1][2]
        y_b = self.position[-1][1] - np.cos(r) * self.center
        y_t = self.position[-1][1] + np.cos(r) * (self.length - self.center)
        self.y_low = min(y_b, y_t)

    def terminated(self):
        return self.y_low < 0

    def get_state(self):
        return np.concatenate([self.position[-1].ravel(), self.v.ravel(), self.omega.ravel()], axis=0)

    def set_state(self, x, y, r, v_x, v_y, w):
        self.position.append(np.array([x, y, r]))
        self.v = np.array([[v_x],
                           [v_y]])
        self.omega = np.array([w])


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
