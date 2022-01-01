import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd

import sys
import os
sys.path.append(os.path.realpath(os.path.dirname(__file__)))
import utils
class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [pyd.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_dim, action_dim, hidden_channels,
                 log_std_bounds):
        super().__init__()

        # self.log_std_bounds = log_std_bounds
        log_std_min, log_std_max = log_std_bounds
        # conversion x:[-1,1]->y:[log_std_min, log_std_max] with y = k*x+b
        self.k = (log_std_max - log_std_min)/2.
        self.b = log_std_min + self.k
        
        self.trunk = utils.mlp(obs_dim, hidden_channels, 2 * action_dim)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        # log_std_min, log_std_max = self.log_std_bounds
        # log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +1)
        log_std = self.k*log_std + self.b

        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)