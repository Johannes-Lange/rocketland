from src.render import Visualization
from src.rocket import Rocket
from random import uniform as uf
from src.tools import normalize_state
from copy import deepcopy
import torch.nn as nn
import numpy as np
import torch
import random
import pickle
import math
from config import *


class DQL:
    def __init__(self, vis=True):
        self.w, self.h = W, H
        self.vis = vis

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        print(self.device)

        self.memory = Memory(self.device)
        self.score = []

        # hyperparameter dql
        self.gamma = 0.99
        self.number_of_samples = 100000

        # physics
        self.rocket = Rocket(self.w, self.h, prop=0.5)

        # optimizer and loss criterion
        self.policy_net = DQLModel()
        self.target_net = DQLModel()
        self.target_net.eval()
        if self.cuda:
            self.policy_net.layers = self.policy_net.layers.to(self.device)
            self.target_net.layers = self.target_net.layers.to(self.device)
        self.target_net.layers.load_state_dict(self.policy_net.get_state())
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.crit = torch.nn.MSELoss()

        # rendering
        if self.vis:
            self.render = Visualization(600, 760)

    def iteration(self, m):
        scores = []
        for game in range(m):
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * game / EPS_DECAY)

            # randomly set state_0, reset score
            score = 0
            self.rocket.set_state(uf(-100, 100), uf(650, 750), uf(-0.2, 0.2), uf(-20, 20), uf(-120, -80), uf(-0.2, 0.2))
            state_0 = torch.tensor(self.rocket.get_state(), device=self.device).float()
            while not self.rocket.dead:
                # select next action
                if random.random() < eps_threshold:
                    action_idx = random.randint(0, len(ACTION)-1)
                else:
                    with torch.no_grad():
                        out = self.policy_net(state_0)
                    action_idx = torch.max(out, dim=0)[1].item()

                # execute action (state_0, action, reward, state_1, terminal)
                action = ACTION[action_idx]  # to feed the rocket
                transition = self.rocket.update(*action)
                score += transition[2]

                if self.vis:
                    if game % 50 == 0:
                        self.render.frame(self.rocket, transition[2], realtime=True)
                    else:
                        self.render.clear(game)

                # keep transition in memory (state_0, action, reward, state_1, terminal)
                self.memory.push(transition, action_idx)

                # train minibatch
                self.train()
                state_0 = torch.tensor(transition[3], device=self.device).float()

            self.score.append(score)
            print('game: {} score: {:.2f} mean: {:.2f} eps: {:.2f}'.format(game, score,
                                                                           sum(self.score) / len(self.score),
                                                                           eps_threshold))
            # save state, update target net weights
            if (game+1) % 1000 == 0:
                self.policy_net.save_state()
                pickle.dump(scores, open('runs/scores.pkl', 'wb'))
            if game % 10 == 0:
                self.target_net.layers.load_state_dict(deepcopy(self.policy_net.get_state()))

    def train(self):
        # get batch of transitions
        state_0_batch, action_batch, reward_batch, state_1_batch, terminated = self.memory.get_batch()

        # compute q_values
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

    def test(self, state_dict=None):
        if state_dict is not None:
            self.policy_net.layers.load_state_dict(state_dict)
        for _ in range(10):
            self.rocket.set_state(uf(-100, 100), uf(650, 750), uf(-0.2, 0.2), uf(-20, 20), uf(-120, -80), uf(-0.2, 0.2))
            while not self.rocket.dead:
                state_0 = torch.tensor(self.rocket.get_state(), device=self.device).float()
                # select next action
                with torch.no_grad():
                    out = self.policy_net(state_0)
                action_idx = torch.max(out, dim=0)[1].item()

                # execute action (state_0, action, reward, state_1, terminal)
                action = ACTION[action_idx]  # to feed the rocket
                transition = self.rocket.update(*action)
                self.render.frame(self.rocket, transition[2], realtime=True)


class DQLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('input', nn.Linear(7, 128))  # state is 7 dim
        self.layers.add_module('ai', nn.Tanh())
        self.layers.add_module('h1', nn.Linear(128, 128))
        self.layers.add_module('a1', nn.ReLU())
        # self.layers.add_module('h2', nn.Linear(128, 128))
        # self.layers.add_module('a2', nn.ReLU())
        self.layers.add_module('h3', nn.Linear(128, 90))
        self.layers.add_module('a3', nn.ReLU())
        self.layers.add_module('h4', nn.Linear(90, 70))
        self.layers.add_module('a4', nn.ReLU())
        self.layers.add_module('out', nn.Linear(70, len(ACTION)))

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

    def get_state(self):
        return self.layers.state_dict()

    def save_state(self):
        state = self.layers.state_dict()
        pickle.dump(state, open('runs/state.pkl', 'wb'))

    def load_state(self):
        state = pickle.load(open('runs/state.pkl', 'rb'))
        self.layers.load_state_dict(state)


class Memory:
    def __init__(self, device):
        self.max_samp = MEMORY_SZ
        self.batch_sz = MINIBATCH
        self.memory = []
        self.device = device

    def push(self, transitition, action_idx):
        action_vec = torch.zeros(len(ACTION))
        action_vec[action_idx] = 1

        state_0, _, reward, state_1, termination = transitition
        save = (state_0, action_vec, reward, state_1, termination)

        self.memory.append(save)
        if len(self.memory) > self.max_samp:
            self.memory.pop(0)

    def get_batch(self):
        batch = random.sample(self.memory, min(self.batch_sz, len(self.memory)))

        state_0_batch, state_1_batch, action_batch, reward_batch = [], [], [], []
        terminated = []
        for t in batch:
            # norm state
            s0, s1 = t[0], t[3]
            state_0 = normalize_state(t[0])
            state_1 = normalize_state(t[3])
            # for j in range(s0.shape[0]):
            #     s0[j] = norm_minmax(s0[j], STATE_MINMAX[j])
            #     s1[j] = norm_minmax(s1[j], STATE_MINMAX[j])
            state_0_batch.append(state_0)
            state_1_batch.append(state_1)
            action_batch.append(t[1])
            reward_batch.append(t[2])
            terminated.append(t[-1])
        state_0_batch = torch.tensor(state_0_batch, device=self.device).float()
        action_batch = torch.stack(action_batch).float().to(self.device)
        state_1_batch = torch.tensor(state_1_batch, device=self.device).float()
        reward_batch = torch.tensor(reward_batch, device=self.device).float()

        return state_0_batch, action_batch, reward_batch, state_1_batch, terminated


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0)
