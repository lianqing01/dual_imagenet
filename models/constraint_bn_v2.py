from __future__ import division

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn

class Constraint_Norm(nn.Module):

    def __init__(self, num_features):
        super(Constraint_Norm, self).__init__()
        self.num_features = num_features
        self.set_dim()
        self.mu_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.gamma_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))

        #initialization
        self.mu_.data.fill_(0)
        self.gamma_.data.fill_(1)
        self.lagrangian = Constraint_Lagrangian(num_features)

        # strore mean and variance for reference
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.zeros(num_features))
        self.register_buffer("tracking_times", torch.tensor(0, dtype=torch.long))

    def get_mean_var(self):
        return (self.mean.abs().sum() / (self.tracking_times + 1e-11), self.var.abs().sum() / (self.tracking_times + 1e-11))

    def set_dim(self):
        raise NotImplementedError



    def forward(self, x):

        # mean
        x = x - self.mu_
        mean = self.lagrangian.get_weighted_mean(x, self.norm_dim)
        self.mean += mean.detach()

        # var
        x = self.gamma_ * x
        var = self.lagrangian.get_weighted_var(x, self.norm_dim)
        self.var += var.detach()

        self.tracking_times += 1
        return x



    def reset_norm_statistics(self):
        self.mean.fill_(0)
        self.var.fill_(0)
        self.tracking_times.fill_(0)



class Constraint_Norm1d(Constraint_Norm):
    def __init__(self, num_features):
        super(Constraint_Norm1d, self).__init__(num_features)

    def set_dim(self):
        self.feature_dim = [1, self.num_features]
        self.norm_dim = [0]

class Constraint_Norm2d(Constraint_Norm):
    def __init__(self, num_features):
        super(Constraint_Norm2d, self).__init__(num_features)

    def set_dim(self):
        self.feature_dim = [1, self.num_features, 1, 1]
        self.norm_dim = [0, 2, 3]




class Constraint_Lagrangian(nn.Module):

    def __init__(self, num_features):
        super(Constraint_Lagrangian, self).__init__()
        self.num_features = num_features
        self.lambda_ = nn.Parameter(torch.Tensor(num_features))
        self.xi_ = nn.Parameter(torch.Tensor(num_features))
        self.lambda_.data.fill_(0)
        self.xi_.data.fill_(0)

    def get_weighted_mean(self, x, norm_dim):
        x_ = x.clone()
        mean = x.mean(dim=norm_dim)
        self.weight_mean = self.xi_ * mean
        self.weight_mean = self.weight_mean.sum()
        return mean

    def get_weighted_var(self, x, norm_dim):
        x_ = x.clone()
        var = x**2
        var = var.mean(dim=norm_dim)
        self.weight_var = self.lambda_ * var
        self.weight_var = self.weight_var.sum()
        return var
    def get_weight_mean_var(self):
        return (self.weight_mean, self.weight_var)


class Constraint_Affine(nn.Module):
    def __init__(self, num_features):
        super(Constraint_Affine, self).__init__()
        self.num_features = num_features
        self.set_dim()

        self.c_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.u_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.c_.data.fill_(0)
        self.u_.data.fill_(1)

    def set_dim(self):
        raise NotImplementedError


    def forward(self, x):
        return x * self.u_ + self.c_

class Constraint_Affine1d(Constraint_Affine):
    def __init__(self, num_features):
        super(Constraint_Affine1d, self).__init__(num_features)

    def set_dim(self):
        self.feature_dim = [1, self.num_features]


class Constraint_Affine2d(Constraint_Affine):
    def __init__(self, num_features):
        super(Constraint_Affine2d, self).__init__(num_features)

    def set_dim(self):
        self.feature_dim = [1, self.num_features, 1, 1]


