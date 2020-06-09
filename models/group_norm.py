import torch
import torch.nn as nn

from torch.nn import Parameter


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps
        self.sample_noise = False
        self.num_features = num_features

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0
        biased = H * W
        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = (x*x).mean(-1, keepdim=True) - mean ** 2
        var *= (H*W*G - 1) / (H*H*G)
        var = torch.clamp(var, min=0)
        if self.sample_noise and self.training:
            # noise MEAN
            noise_mean = torch.normal(mean=self.noise_mean, std=self.noise_mean_std).detach().clamp(min=0)
            # noise VAR
            noise_var = torch.normal(mean=self.noise_mean, std=self.noise_var_std).detach().clamp(min=0)

            mean = mean * noise_mean.view(mean.size())
            var = var * noise_var.view(mean.size())

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

