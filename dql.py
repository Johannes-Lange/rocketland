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

p = [0., .5, .6, .7, .8, 1.]
a = [-np.pi/6, -np.pi/18, 0, np.pi/18, np.pi/6]
ACTION = list(product(a, p))
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 400
STATE_MINMAX = [(0, 40), (-600 / 2, 600 / 2), (0, 760), (-np.pi/3, np.pi/3), (-40, 40), (-120, 50), (-1.2, 1.2)]


class DQL:
    def __init__(self, vis=True):
        self.w, self.h = 600, 760
        self.vis = vis
        self.memory = []

        # hyperparameter dql
        self.gamma = 0.99
        self.batch_size = 100
        self.nb_batches = 100
        self.number_of_samples = 50000

        # physics
        self.rocket = Rocket(self.w, self.h, prop=0.5)

        # optimizer and loss criterion
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.policy_net = DQLModel()
        self.target_net = DQLModel()
        self.target_net.eval()
        if self.cuda:
            self.policy_net.layers = self.policy_net.layers.to(self.device)
            self.target_net.layers = self.target_net.layers.to(self.device)
        self.target_net.layers.load_state_dict(self.policy_net.get_state())
        self.optim = torch.optim.SGD(self.policy_net.parameters(), lr=1e-6)
        self.crit = torch.nn.SmoothL1Loss()

        # rendering
        if self.vis:
            self.render = Visualization(600, 760)

    def iteration(self, m):
        scores = []
        for game in range(m):
            # randomly set state_0, reset score and lifespan
            self.rocket.set_state(uf(-100, 100), uf(650, 750), uf(-0.2, 0.2), uf(-20, 20), uf(-120, -80), uf(-0.2, 0.2))

            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * game / EPS_DECAY)

            score = 0
            while not self.rocket.dead:
                state_0 = torch.tensor(self.rocket.get_state(), device=self.device).float()

                # action
                if random.random() < eps_threshold:
                    action_idx = random.randint(0, len(ACTION)-1)
                else:
                    with torch.no_grad():
                        out = self.policy_net(state_0)
                    action_idx = torch.max(out, dim=0)[1].item()
                action = ACTION[action_idx]  # to feed the rocket

                # execute action (state_0, action, reward, state_1, terminal)
                state_0, _, reward, state_1, terminal = self.rocket.update(*action)
                score += reward
                if self.vis:
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

            # sample random minibatches of size batch
            for i in range(self.nb_batches):
                batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))

                state_0_batch, state_1_batch, action_batch, reward_batch = [], [], [], []
                terminated = []
                for t in batch:
                    # norm state
                    s0, s1 = t[0], t[3]
                    for j in range(s0.shape[0]):
                        s0[j] = norm_minmax(s0[j], STATE_MINMAX[j])
                        s1[j] = norm_minmax(s1[j], STATE_MINMAX[j])
                    state_0_batch.append(s0)
                    state_1_batch.append(s1)
                    action_batch.append(t[1])
                    reward_batch.append(t[2])
                    terminated.append(t[-1])
                state_0_batch = torch.tensor(state_0_batch, device=self.device).float()
                action_batch = torch.stack(action_batch).float().to(self.device)
                state_1_batch = torch.tensor(state_1_batch, device=self.device).float()
                reward_batch = torch.tensor(reward_batch, device=self.device).float()

                q_eval = torch.sum(self.policy_net(state_0_batch) * action_batch, dim=1).float()
                with torch.no_grad():
                    q_next = self.target_net(state_1_batch)
                q_next[terminated] = 0.
                q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0].detach()

                # learn
                self.optim.zero_grad()
                loss = self.crit(q_target, q_eval)
                loss.backward()
                nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
                self.optim.step()

            scores.append(score)
            print('game: {} score: {:.2f} mean: {:.2f} eps: {:.2f}'.format(game, score,
                                                                           sum(scores) / len(scores), eps_threshold))
            if (game+1) % 1000 == 0:
                self.policy_net.save_state()
                pickle.dump(scores, open('scores.pkl', 'wb'))
            if game % 10 == 0:
                self.target_net.layers.load_state_dict(self.policy_net.get_state())


class DQLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('input', nn.Linear(7, 100))  # state is 7 dim
        self.layers.add_module('ai', nn.Tanh())
        self.layers.add_module('h1', nn.Linear(100, 80))
        self.layers.add_module('a1', nn.ReLU())
        self.layers.add_module('h2', nn.Linear(80, 60))
        self.layers.add_module('a2', nn.ReLU())
        self.layers.add_module('out', nn.Linear(60, len(ACTION)))

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

    def get_state(self):
        return self.layers.state_dict()

    def save_state(self):
        state = self.layers.state_dict()
        pickle.dump(state, open('state.pkl', 'wb'))

    def load_state(self):
        state = pickle.load(open('state.pkl', 'rb'))
        self.layers.load_state_dict(state)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0)


def norm_minmax(val, minmax):
    low, high = minmax[0], minmax[1]
    if low < 0:
        val += abs(low)
        high += abs(low)
        low += abs(low)
    return (val - low) / (high - low)
