import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from functools import reduce
from .config import devices


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        batch_size = obs.size()[0]
        a = self.pi(obs.reshape([batch_size, -1]))
        return self.act_limit * (a.cpu() if devices.type == 'cuda' else a)


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        batch_size = obs.size()[0]
        obs = obs.reshape([batch_size, -1])
        act = act.reshape([batch_size, -1])
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = reduce(lambda x, y: x * y, observation_space.shape)
        act_dim = reduce(lambda x, y: x * y, action_space.shape)
        act_limit = action_space.high[0]

        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        batch_size = obs.size()[0]
        with torch.no_grad():
            a = self.pi(obs.reshape([batch_size, -1]))
            return a.cpu().numpy() if devices.type == 'cuda' else a.numpy()
