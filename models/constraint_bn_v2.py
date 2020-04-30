from __future__ import division

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.autograd import Function

class LagrangianFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        # input shape: [1, C, 1, 1]
        # weight shape: [1, C, 1, 1]
        # output shape: [1, C, 1, 1]
        ctx.save_for_backward(input, weight)
        output = input * weight
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight
        if ctx.needs_input_grad[1]:
            # gradient ascent
            grad_weight = -1 * grad_output * input
        return grad_input, grad_weight


class Constraint_Norm(nn.Module):

    def __init__(self, num_features, weight_decay=1e-3, get_optimal_lagrangian=False, pre_affine=True, post_affine=True):
        super(Constraint_Norm, self).__init__()
        self.num_features = num_features
        self.pre_affine=pre_affine
        self.post_affine = post_affine
        self.set_dim()
        self.mu_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))
        self.gamma_ = nn.Parameter(torch.Tensor(num_features).view(self.feature_dim))

        #initialization
        self.mu_.data.fill_(0)
        self.gamma_.data.fill_(1)
        self.lagrangian = Constraint_Lagrangian(num_features,
                                                weight_decay=weight_decay,
                                                get_optimal_lagrangian=get_optimal_lagrangian)

        # strore mean and variance for reference
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.zeros(num_features))
        self.register_buffer("tracking_times", torch.tensor(0, dtype=torch.long))
        self.update_affine_only = False

    def get_mean_var(self):
        with torch.no_grad():
            mean = self.mean / (self.tracking_times + 1e-11)
            var = self.var / (self.tracking_times + 1e-11)
            mean = mean.abs().mean()
            var = var.abs().mean()
        return mean, var

    def set_dim(self):
        raise NotImplementedError



    def forward(self, x):

        # mean
        if self.pre_affine:
            if self.update_affine_only:
                x_ = x.detach() - self.mu_
                mean = self.lagrangian.get_weighted_mean(x_, self.norm_dim)
                x = x - self.mu_.detach()
            else:
                x = x - self.mu_
                mean = self.lagrangian.get_weighted_mean(x, self.norm_dim)
        else:
            mean = self.lagrangian.get_weighted_mean(x, self.norm_dim)
        try:
            self.mean += mean.detach()
        except:
            import pdb
            pdb.set_trace()
        # var
        if self.pre_affine:
            if self.update_affine_only:
                x_ = x.detach() * self.gamma_
                var = self.lagrangian.get_weighted_var(x_, self.norm_dim)
                x = x * self.gamma_.detach()
            else:
                x = x * self.gamma_
                var = self.lagrangian.get_weighted_var(x, self.norm_dim)
        else:
            var = self.lagrangian.get_weighted_var(x, self.norm_dim)
        self.var += var.detach()

        self.tracking_times += 1
        if self.post_affine != False:
            x = self.post_affine_layer(x)
        return x



    def reset_norm_statistics(self):
        self.mean.fill_(0)
        self.var.fill_(0)
        self.tracking_times.fill_(0)



class Constraint_Norm1d(Constraint_Norm):
    def __init__(self, num_features, pre_affine=True, post_affine=True):
        super(Constraint_Norm1d, self).__init__(num_features, pre_affine=pre_affine, post_affine=post_affine)

    def set_dim(self):
        self.feature_dim = [1, self.num_features]
        self.norm_dim = [0]
        if self.post_affine != False:
            self.post_affine_layer = Constraint_Affine1d(self.num_features)

class Constraint_Norm2d(Constraint_Norm):
    def __init__(self, num_features, pre_affine=True, post_affine=True):
        super(Constraint_Norm2d, self).__init__(num_features, pre_affine=pre_affine, post_affine=post_affine)

    def set_dim(self):
        self.feature_dim = [1, self.num_features, 1, 1]
        self.norm_dim = [0, 2, 3]
        if self.post_affine != False:
            self.post_affine_layer = Constraint_Affine2d(self.num_features)




class Constraint_Lagrangian(nn.Module):

    def __init__(self, num_features, weight_decay=1e-4, get_optimal_lagrangian=False):
        super(Constraint_Lagrangian, self).__init__()
        self.num_features = num_features
        self.lambda_ = nn.Parameter(torch.Tensor(num_features))
        self.xi_ = nn.Parameter(torch.Tensor(num_features))
        self.lambda_.data.fill_(0)
        self.xi_.data.fill_(0)
        self.weight_decay = weight_decay
        self.get_optimal_lagrangian = get_optimal_lagrangian

    def get_weighted_mean(self, x, norm_dim):
        mean = x.mean(dim=norm_dim)
        self.weight_mean = LagrangianFunction.apply(mean, self.xi_)
        self.weight_mean = self.weight_mean.sum()
        self.weight_mean_abs = self.weight_mean.abs().sum().detach()
        return mean

    def get_weighted_var(self, x, norm_dim):
        var = x**2 - 1
        var = var.mean(dim=norm_dim)
        self.weight_var = LagrangianFunction.apply(var, self.lambda_)
        self.weight_var = self.weight_var.sum()
        self.weight_var_abs = self.weight_var.abs().sum().detach()
        return var
    def get_weight_mean_var(self):
        return (self.weight_mean, self.weight_var)

    def get_weight_mean_var_abs(self):
        return (self.weight_mean_abs, self.weight_var_abs)


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


