from src.render import Visualization
from src.rocket import Rocket
from random import uniform as uf
from datetime import datetime
from copy import deepcopy
import torch.nn as nn
import torch
import random
import pickle
import math
from config import *


class DQL:
    def __init__(self, vis=True):
        self.date = datetime.now().strftime('%Y%m%d%H%M')
        self.w, self.h = W, H
        self.vis = vis

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        print(self.device)

        self.memory = Memory(self.device)
        self.score = []

        # hyperparameter dql
        self.gamma = 0.99

        # physics
        self.rocket = Rocket(self.w, self.h, prop=0.5)

        # optimizer and loss criterion
        self.policy_net = DQLModel(self.date)
        self.target_net = DQLModel(self.date)
        self.target_net.eval()
        if self.cuda:
            self.policy_net.layers = self.policy_net.layers.to(self.device)
            self.target_net.layers = self.target_net.layers.to(self.device)
        self.target_net.layers.load_state_dict(self.policy_net.get_state())
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.crit = torch.nn.MSELoss()

        # rendering
        if self.vis:
            self.render = Visualization(600, 760)

    def iteration(self, m):
        # keep stats
        losses = []
        scores = []

        for game in range(m):
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * game / EPS_DECAY)

            # randomly set state_0, reset score
            game_duration = 0
            score = 0
            loss = 0
            self.rocket.set_state(uf(-100, 100), uf(650, 750), uf(-2.5, 2.5), uf(-20, 20), uf(-120, -80), uf(-0.6, 0.6))
            state_0 = torch.tensor(self.rocket.get_state(), device=self.device).float()
            while not self.rocket.dead:
                game_duration += 1
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
                loss += self.train()
                state_0 = torch.tensor(transition[3], device=self.device).float()

            # stats
            losses.append(loss / game_duration)
            scores.append(score)

            mean_loss = sum(scores[-100:]) / len(scores[-100:])
            print('game: {} score: {:.2f} mean: {:.2f} loss: {:.3f} eps: {:.2f}'.format(game, score, mean_loss,
                                                                                        losses[-1], eps_threshold))

            # save state, update target net weights
            if (game+1) % 100 == 0:
                dump = {'scores': scores, 'losses': losses}
                self.policy_net.save_state()
                # pickle.dump(dump, open('runs/' + self.date + '_stats.pkl', 'wb'))
            if game % TARGET_UPDATE == 0:
                self.target_net.load_state(deepcopy(self.policy_net.get_state()))

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
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optim.step()
        return loss.item()/state_0_batch.shape[0]

    def test(self, state_dict=None):
        if state_dict is not None:
            print('load')
            self.policy_net.load_state(state_dict)
        for _ in range(50):
            self.rocket.set_state(uf(-100, 100), uf(650, 750), uf(-2.5, 2.5), uf(-20, 20), uf(-120, -80), uf(-0.6, 0.6))
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
    def __init__(self, date):
        super().__init__()
        self.date = date

        self.input_sz = 7  # state dim = 7
        self.output_sz = len(ACTION)

        self.layers = nn.Sequential()
        self.layers.add_module('input', nn.Linear(self.input_sz, 128))

        self.layers.add_module('h1', nn.Linear(128, 128))  # (100, 200)
        self.layers.add_module('a1', nn.Sigmoid())

        self.layers.add_module('h2', nn.Linear(128, 128))
        self.layers.add_module('a2', nn.Sigmoid())

        # self.layers.add_module('h2', nn.Linear(128, 128))
        # self.layers.add_module('a2', nn.ReLU())

        self.layers.add_module('out', nn.Linear(128, self.output_sz))

        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

    def get_state(self):
        return self.layers.state_dict()

    def save_state(self):
        state = self.layers.state_dict()
        pickle.dump(state, open('runs/' + self.date + '_state.pkl', 'wb'))

    def load_state(self, state):
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
            # normalize both states same as in reward function
            # bulid batches for training
            state_0_batch.append(t[0])
            action_batch.append(t[1])
            reward_batch.append(t[2])
            state_1_batch.append(t[3])
            terminated.append(t[4])
        state_0_batch = torch.tensor(state_0_batch, device=self.device).float()
        action_batch = torch.stack(action_batch).float().to(self.device)
        state_1_batch = torch.tensor(state_1_batch, device=self.device).float()
        reward_batch = torch.tensor(reward_batch, device=self.device).float()

        return state_0_batch, action_batch, reward_batch, state_1_batch, terminated


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)
