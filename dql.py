from rocket import Rocket
from render import Visualization
import numpy as np
from random import uniform as uf
import random
import pickle
import math
import torch
import torch.nn as nn
from itertools import product

p = [i/10 for i in range(3, 11)]
a = [-0.2, -0.05, -0.01, 0, 0.01, 0.05, 0.2]
ACTION = list(product(a, p))
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 400
STATE_MINMAX = [(-600 / 2, 600 / 2), (0, 760), (-np.pi/3, np.pi/3), (-40, 40), (-120, 50), (-1.2, 1.2)]


class DQL:
    def __init__(self):
        self.w, self.h = 600, 760
        self.number_of_samples = 50000
        self.memory = []

        self.gamma = 0.99
        self.batch_size = 5000

        self.rocket = Rocket(self.w, self.h, prop=0.5)

        # optimizer and loss criterion
        self.model = DQLModel()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.crit = torch.nn.SmoothL1Loss()

        self.render = Visualization(600, 760)

    def iteration(self, m):
        scores = []
        for game in range(m):
            # randomly set state_0, reset score and lifespan
            self.rocket.set_state(uf(-50, 50), uf(650, 750), uf(-0.2, 0.2), uf(-20, 20), uf(-120, -80), 0)

            score = 0
            while not self.rocket.dead:
                state_0 = torch.tensor(self.rocket.get_state()).float()

                # action
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * game / EPS_DECAY)
                if random.random() < eps_threshold:
                    action_idx = random.randint(0, len(ACTION)-1)
                else:
                    out = self.model(state_0)
                    action_idx = torch.max(out, dim=0)[1].item()
                action = ACTION[action_idx]  # to feed the rocket

                # execute action (state_0, action, reward, state_1, terminal)
                state_0, _, reward, state_1, terminal = self.rocket.update(*action)
                score += reward
                print(reward)
                if game % 50 == 0:
                    self.render.frame(self.rocket, reward, realtime=True)
                else:
                    self.render.clear(game)

                # keep transition in memory (state_0, action, reward, state_1, terminal)
                action_vec = torch.zeros(len(ACTION))
                action_vec[action_idx] = 1
                self.memory.append((state_0, action_vec, reward, state_1, terminal))
                if len(self.memory) > self.number_of_samples:
                    self.memory.pop(0)

            # sample random minibatch of size x
            batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))

            state_0_batch, state_1_batch, action_batch, reward_batch = [], [], [], []
            terminated = []
            for t in batch:
                # norm state
                s0, s1 = t[0], t[3]
                for i in range(s0.shape[0]):
                    s0[i] = norm_minmax(s0[i], STATE_MINMAX[i])
                    s1[i] = norm_minmax(s1[i], STATE_MINMAX[i])
                state_0_batch.append(s0)
                state_1_batch.append(s1)
                action_batch.append(t[1])
                reward_batch.append(t[2])
                terminated.append(t[-1])
            state_0_batch = torch.tensor(state_0_batch).float()
            action_batch = torch.stack(action_batch).float()
            state_1_batch = torch.tensor(state_1_batch).float()
            reward_batch = torch.tensor(reward_batch).float()

            q_eval = torch.sum(self.model(state_0_batch) * action_batch, dim=1).float()
            q_next = self.model(state_1_batch)
            q_next[terminated] = 0.
            q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

            # learn
            self.optim.zero_grad()
            loss = self.crit(q_target, q_eval)
            loss.backward()
            self.optim.step()

            scores.append(score)
            if game+1 % 1000 == 0:
                self.model.save_state()
                pickle.dump(scores, open('scores.pkl', 'wb'))


class DQLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('input', nn.Linear(6, 100))  # state is 7 dim
        self.layers.add_module('ai', nn.Tanh())
        self.layers.add_module('h1', nn.Linear(100, 80))
        self.layers.add_module('a1', nn.ReLU())
        self.layers.add_module('h2', nn.Linear(80, 60))
        self.layers.add_module('a2', nn.ReLU())
        self.layers.add_module('out', nn.Linear(60, len(ACTION)))

    def forward(self, x):
        return self.layers(x)

    def save_state(self):
        state = self.layers.state_dict()
        pickle.dump(state, open('state.pkl', 'wb'))

    def load_state(self):
        state = pickle.load(open('state.pkl', 'rb'))
        self.layers.load_state_dict(state)


def norm_minmax(val, minmax):
    low, high = minmax[0], minmax[1]
    if low < 0:
        val += abs(low)
        high += abs(low)
        low += abs(low)
    return (val - low) / (high - low)
