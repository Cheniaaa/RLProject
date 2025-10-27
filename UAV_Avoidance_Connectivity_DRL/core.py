import random
from collections import deque

from torch import nn

from config import Config


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        layers = []
        prev_dim = state_dim
        for mid_dim in Config.VALUE_NET_HIDDEN:
            layers.append(nn.Linear(prev_dim, mid_dim))
            layers.append(nn.ReLU())
            prev_dim = mid_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, s):
        return self.network(s)


class SINRNetwork(nn.Module):
    def __init__(self, state_dim):
        super(SINRNetwork, self).__init__()
        layers = []
        prev_dim = state_dim
        for mid_dim in Config.SINR_NET_HIDDEN:
            layers.append(nn.Linear(prev_dim, mid_dim))
            layers.append(nn.ReLU())
            prev_dim = mid_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, s):
        return self.network(s)


class ReplayBuffer:
    def __init__(self, max_capacity):
        self.buffer = deque(maxlen=max_capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        """随机采样一批经验"""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
