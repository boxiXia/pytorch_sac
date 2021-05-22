import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import gym
import os
from collections import deque
import random
import math

# import dmc2gym
# def makeEnv(cfg):
#     """Helper function to create dm_control environment"""
#     if cfg.env == 'ball_in_cup_catch':
#         domain_name = 'ball_in_cup'
#         task_name = 'catch'
#     else:
#         domain_name = cfg.env.split('_')[0]
#         task_name = '_'.join(cfg.env.split('_')[1:])
#     
#     env = dmc2gym.make(domain_name=domain_name,
#                        task_name=task_name,
#                        seed=cfg.seed,
#                        visualize_reward=True)
#     env.seed(cfg.seed)
#     assert env.action_space.low.min() >= -1
#     assert env.action_space.high.max() <= 1
#     return env


class evalMode(object):
    """ context manager that sets the module in evaluation mode."""
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class trainMode(object):
    """ context manager that sets the module in training mode."""
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

class evalMode(trainMode):
    """ context manager that sets the module in evaluation mode."""


def softUpdateParams(net, target_net, tau):
    """target_net = net*tau + target_net*(1-tau)"""
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def setSeedEverywhere(seed):
    """set seed on pytroch, numpy.random and python random """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def makeDir(*path_parts):
    """create directory give path parts"""
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    """return dense layers"""
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

####################################################################################################
# reference: 
# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
# https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU()],
        ['selu', nn.SELU()],
        ['none', nn.Identity()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
        self.should_apply_shortcut = (self.in_channels != self.out_channels)

    def forward(self, x):
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        else:
            residual = x
        x = self.blocks(x)
        x += residual
        return x

# K. He, X. Zhang, S. Ren, and J. Sun. Identity Mappings in Deep Residual Networks. arXiv preprint arXiv:1603.05027v3,2016.
class ResidualDenseBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels,seq_len=None, activation='leaky_relu'):
        super().__init__(in_channels, out_channels)
        self.blocks = nn.Sequential(
            # nn.BatchNorm1d(in_channels if seq_len is None else seq_len),
            # activation_func(activation),
            nn.Linear(in_channels, out_channels),
            # nn.BatchNorm1d(out_channels if seq_len is None else seq_len),
            activation_func(activation),
            nn.Linear(out_channels, out_channels),
            activation_func(activation)
        )
        self.shortcut = nn.Linear(in_channels, out_channels)
##########################################################################################################################



# def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
#     if hidden_depth == 0:
#         mods = [nn.Linear(input_dim, output_dim)]
#     else:
#         mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
#         for i in range(hidden_depth - 1):
#             mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
#         mods.append(nn.Linear(hidden_dim, output_dim))
#     if output_mod is not None:
#         mods.append(output_mod)
#     trunk = nn.Sequential(*mods)
#     return trunk

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [ResidualDenseBlock(input_dim,hidden_dim, activation="leaky_relu")]
        for i in range(hidden_depth - 1):
            mods += [ResidualDenseBlock(hidden_dim,hidden_dim, activation="leaky_relu")]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk



def toNumpy(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.detach().cpu().numpy()
