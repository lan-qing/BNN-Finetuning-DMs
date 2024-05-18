import math
import numbers
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .weight_noise import noise_fn


class RandGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, init_s=-1e10, eps=1e-5, affine=True):
        super(RandGroupNorm, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.init_s = init_s
        if self.affine:
            self.mu_weight = Parameter(torch.empty(num_channels))
            self.sigma_weight = Parameter(torch.empty(num_channels))
            self.mu_bias = Parameter(torch.empty(num_channels))
            self.sigma_bias = Parameter(torch.empty(num_channels))
        else:
            self.register_parameter('mu_weight', None)
            self.register_parameter('mu_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            self.mu_weight.data.uniform_()
            self.sigma_weight.data.fill_(self.init_s)
            self.mu_bias.data.zero_()
            self.sigma_bias.data.fill_(self.init_s)

    def forward(self, x, sample=True, scale=0.0):
        if self.affine:
            with torch.no_grad():
                self.sigma_weight.data = torch.maximum(self.sigma_weight, torch.ones_like(self.sigma_weight) * 1e-7).type(
                    self.sigma_weight.dtype).data
                self.sigma_bias.data = torch.maximum(self.sigma_bias, torch.ones_like(self.sigma_bias) * 1e-7).type(
                    self.sigma_bias.dtype).data
        if not sample or not self.affine:
            weight = self.mu_weight
            bias = self.mu_bias
        else:
            eps_weight = torch.ones(self.sigma_weight.size(), device=x.device).normal_().type(
                self.sigma_weight.dtype)
            weight = self.mu_weight + self.sigma_weight * eps_weight
            eps_bias = torch.ones(self.sigma_bias.size(), device=x.device).normal_().type(self.sigma_bias.dtype)
            bias = self.mu_bias + self.sigma_bias * eps_bias
        return F.group_norm(x, self.num_groups, weight, bias, self.eps)


if __name__ == '__main__':
    pass
