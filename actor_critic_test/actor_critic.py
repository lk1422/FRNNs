import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
from torch import optim
import numpy as np
import numpy as np
import math
import gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_space, n_actions):
        super(PolicyNetwork, self).__init__()
        self.base = nn.Linear(state_space, 256)
        self.actions = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)
        self.rewards = []
        self.action_pairs = [] #[(LOG_PROB, CRITIC_VALUE)]
    def forward(self, x):
        x = F.relu(self.base(x))
        a = F.softmax(self.actions(x), dim=-1)
        v = self.value(x)
        return a,v
    def select_action(self, state):
        a, v = self.forward(state)
        m = Categorical(a)
        action = m.sample()
        self.action_pairs.append((m.log_prob(action), v))
        return action.item()
    def train(self, OPTIM, gamma):
        ##CONSTRUCT SAMPLED VALUES##
        rewards = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        ##NORMALIZE SAMPLED STATE VALUES##
        rewards = torch.tensor(rewards, requires_grad=False)
        rewards = (rewards-rewards.mean())/(rewards.std() + 1e-4)
        ##GET ACTOR AND CRITIC LOSS##
        actor_loss = torch.tensor([0], dtype=torch.float32)
        critic_loss = torch.tensor([0], dtype=torch.float32)
        for (log_prob, val), R in zip(self.action_pairs, rewards):
            advantage = R.item() - val.item()
            actor_loss += -log_prob*advantage
            critic_loss += F.smooth_l1_loss(val, torch.tensor([[R]]))
        total_loss = actor_loss + critic_loss
        ##OPTIMIZE##
        OPTIM.zero_grad()
        total_loss.backward()
        OPTIM.step()
        ##CLEAR MEMORY##
        del self.rewards[:]
        del self.action_pairs[:]


