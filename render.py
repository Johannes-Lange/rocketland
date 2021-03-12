from graphics import *
import numpy as np
import time


class Visualization:
    def __init__(self, w, h):
        self.rocket_len, self.rocket_r, self.dt, self.max_f = None, None, None, None
        self.to_draw = []
        self.last_step = 0
        self.w, self.h = w, h

        # initialize window
        self.window = GraphWin("Rocket", w, h, autoflush=False)
        self.window.setCoords(-w / 2, 0-35, w / 2, h-35)
        self.window.setBackground('black')

        # draw ground level
        self.groundlevel = Line(Point(-w/2, 0), Point(w/2, 0))
        self.groundlevel.setFill('white')
        self.groundlevel.draw(self.window)

        # set text for log
        self.label = Text(Point(-3/8*self.w, self.h * 7 / 8), ' ')
        self.label.setFill('white')
        self.label.setSize(10)
        self.label.setFace('times roman')
        self.label.draw(self.window)

    def frame(self, rocket, score, realtime=True):
        if self.rocket_len is None:
            self.rocket_len = rocket.length
            self.rocket_r = rocket.radius
            self.dt = rocket.t_step
            self.max_f = rocket.f_0
        self.clear()

        self.to_draw = []

        x, y, r = rocket.position[-1][0], rocket.position[-1][1], rocket.position[-1][2]
        x_c = rocket.center
        x_p = rocket.x_prop

        rot_inv = np.linalg.inv(rocket.rotation_matrix())

        r_u = 0.5 * rot_inv.dot(np.array([[self.rocket_r], [0]]))

        center = Circle(Point(x, y), 3)
        center.setFill('red')

        bottom = Point(x + np.sin(r) * x_c - r_u[0], y - np.cos(r) * x_c - r_u[1])
        top = Point(x - np.sin(r) * (self.rocket_len - x_c) - r_u[0], y + np.cos(r) * (self.rocket_len - x_c) - r_u[1])

        # draw propellant blue, empty body white
        d_prop = rot_inv.dot(np.array([[0], [x_p]]))
        prop_up = Point(bottom.getX() + d_prop[0], bottom.getY() + d_prop[1])

        prop_body = Line(bottom, prop_up)
        prop_body.setFill('blue')
        prop_body.setWidth(3)
        self.to_draw.append(prop_body)

        empty_body = Line(prop_up, top)
        empty_body.setFill('white')
        empty_body.setWidth(3)
        self.to_draw.append(empty_body)

        # draw thrust
        phi, thrust = rocket.thruster[0], rocket.thruster[1]/self.max_f
        r_thrust = 20*thrust * rot_inv.dot(np.array([[thrust*np.sin(phi)], [-thrust*np.cos(phi)]]))
        arrow_end = Point(bottom.getX()+r_thrust[0], bottom.getY()+r_thrust[1])

        arrow = Line(bottom, arrow_end)
        arrow.setArrow('last')
        arrow.setWidth(1)
        arrow.setOutline('green')
        self.to_draw.append(arrow)
        self.to_draw.append(center)

        for item in self.to_draw:
            item.draw(self.window)

        # Text
        log = 'y-velocity: {:.1f} m/s\n' \
              'altitude: {:.1f} m\n' \
              'propellant: {:.1f}%\n' \
              'thrust: {:.1f}%\n' \
              'angle: {:.1f}°\n' \
              'reward: {}'.format(rocket.v[1].item(), y, x_p/self.rocket_len*100, thrust*100, np.rad2deg(phi), score)
        self.label.setText(log)

        # update frame for real-time visualization
        if realtime:
            while 1:
                if time.time() > self.last_step + self.dt:
                    update()
                    self.last_step = time.time()
                    break
        else:
            update(300)

    def clear(self):
        for item in self.to_draw:
            item.undraw()
        self.window.update()
