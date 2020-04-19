from __future__ import division

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn


class DualNorm(nn.Module):

    def __init__(self, num_features):
        super(DualNorm, self).__init__()
        self.num_features = num_features
        self.mu_ = nn.Parameter(torch.Tensor(num_features))
        self.lambda_ = nn.Parameter(torch.Tensor(num_features))

        self.mu_.data.fill_(0)
        self.lambda_.data.fill_(0)
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.zeros(num_features))
        self.register_buffer("tracking_times", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # get mean
        x_ = x.clone()
        mean = x_.mean(dim=[0,2,3])
        var = (x_ ** 2 - 1).mean(dim=[0,2,3])

        self.mean += mean.detach()
        self.var += var.detach()
        if self.training:
            self.tracking_times += 1


            self.weight_mean = (self.mu_ * mean).sum()
            # get variance
            self.weight_var = (self.lambda_ * var).sum()
        return x


    def _reset_mean_var(self):
        self.mean.fill_(0)
        self.var.fill_(0)
        self.tracking_times.fill_(0)

    def _get_mean_var(self):
        mean = self.mean / (self.tracking_times + 1e-7)
        var = self.var / (self.tracking_times + 1e-7)
        return (mean.mean(), var.mean())


    def _get_weight_mean_var(self):
        return self.weight_mean, self.weight_var


class DualAffine(nn.Module):
    def __init__(self, num_features):
        super(DualAffine, self).__init__()
        self.num_features = num_features
        self.u_ = nn.Parameter(torch.Tensor(num_features).view([1, num_features, 1, 1]))
        self.b_ = nn.Parameter(torch.Tensor(num_features).view([1, num_features, 1, 1]))
        self.u_.data.fill_(1)
        self.b_.data.fill_(0)

    def forward(self, x):
        return x * self.u_ + self.b_
