import numpy as np
import sympy as sp
import random


class Rocket:
    def __init__(self, rho, l_r, r):
        # geometry
        self.length = l_r
        self.radius = r
        self.cross_sec = np.pi * self.radius ** 2

        # densitys
        rho_al = 2700  # density aluminum
        self.rho_body = (2 * np.pi * self.radius) * 0.03 * rho_al / self.cross_sec  # 3cm wall thicknes
        self.rho_prop = rho

        # mass, complete rocket is tank
        self.x_prop = self.length  # height of propellant
        self.mass = self.length * self.cross_sec * (self.rho_prop + self.rho_body)
        self.center = self.length/2
        self.moment_inertia = 1/3 * self.mass * self.length**2

        # position and thruster
        self.f_0 = 9 * 51e3  # maximal Thrust in Newton, 9 Merlin C1
        self.prop_consumption = 1e-6  # m^3/N, 170s burntime
        self.position = [np.array([0, 200, 0])]  # [np.zeros(3)]  # coordinates + rotation (x, y, R)
        self.thruster = np.zeros(2)  # angle + force (phi, f) (local coordinates)

        self.t_step = 0.1
        self.omega = np.array([0])
        self.v = np.array([[random.randint(-10, 10)],
                           [random.randint(-60, -40)]])
        self.deq = None
        self.symbolic()  # define differential equation self.deq for movement

    def update(self, angle, power):
        # upadte thruster
        self.update_thrust(angle, power)

        # update propellant and mass
        self.x_prop -= self.prop_consumption * self.thruster[1] / self.cross_sec
        self.mass = self.cross_sec * (self.rho_body * self.length + self.rho_prop * self.x_prop)

        # update center + moment inertia
        self.center_gravity()
        self.moment_of_inertia()

        # update position
        self.movement()

    def movement(self):
        v_x_new, v_y_new, w_new = self.deq(self.center, self.moment_inertia, self.omega.item(), self.v[0].item(),
                                           self.v[1].item(), self.t_step, self.thruster[0], self.thruster[1],
                                           self.mass, self.position[-1][2])
        self.omega = np.array([w_new])
        self.v = np.array([[v_x_new],
                           [v_y_new]])
        last_position = self.position[-1]
        self.position.append(last_position + np.array([v_x_new, v_y_new, w_new])*self.t_step)

    def get_state(self):
        return self.position[-1], self.thruster

    def update_thrust(self, angle, p):
        # throttle down to 20% possible --> p between 0.2 and 1
        # F = mass_flow * velocity (+ (p_exit - p_0) * A_exit)
        f = p*self.f_0

        self.thruster[0] = angle
        if self.x_prop > 0:
            self.thruster[1] = f

    def center_gravity(self):
        m_body = self.rho_body * self.cross_sec * self.length
        m_prop = self.rho_prop * self.cross_sec * self.x_prop
        self.center = (m_body * self.length/2 + m_prop * self.x_prop/2) / (m_body + m_prop)

    def moment_of_inertia(self):
        self.moment_inertia = 1/3*self.cross_sec*(self.rho_body*((self.length-self.center)**3-(-self.center)**3) +
                                                  self.rho_prop*((self.x_prop-self.center)**3-(-self.center)**3))

    def rotation_matrix(self):
        return np.array([[np.cos(self.position[-1][3]), -np.sin(self.position[-1][3])],
                         [np.sin(self.position[-1][3]), np.cos(self.position[-1][3])]])

    def symbolic(self):
        r, v_x, v_x_l, v_y, v_y_l, t, phi, p, m, j_m, w, w_l, x_c = sp.symbols('r v_x v_x_l v_y v_y_l t \
                                                                               phi p m j_m w w_l x_c')
        # rotation matrix
        rot = sp.Matrix([[sp.cos(r), -sp.sin(r)],
                         [sp.sin(r), sp.cos(r)]])
        # gravity
        f_g = rot * sp.Matrix([[0],
                               [-9.81 * m]])
        # thrust
        f_t = sp.Matrix([[-p * sp.sin(phi)],
                         [p * sp.cos(phi)]])
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
        self.deq = sp.lambdify([x_c, j_m, w_l, v_x_l, v_y_l, t, phi, p, m, r], sol)

        print(sol)
