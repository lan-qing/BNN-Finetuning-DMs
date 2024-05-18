import math
import numbers
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .weight_noise import noise_fn


class RandLayerNorm(nn.Module):
    def __init__(self, normalized_shape, init_s=-1e10, eps=1e-5, elementwise_affine=True, bias=True):
        super(RandLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.biasp = bias
        self.init_s = init_s
        if self.elementwise_affine:
            self.mu_weight = Parameter(torch.empty(self.normalized_shape))
            self.sigma_weight = Parameter(torch.empty(self.normalized_shape))
            if bias:
                self.mu_bias = Parameter(torch.empty(self.normalized_shape))
                self.sigma_bias = Parameter(torch.empty(self.normalized_shape))
            else:
                self.register_parameter('mu_bias', None)
        else:
            self.register_parameter('mu_weight', None)
            self.register_parameter('mu_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            self.mu_weight.data.uniform_()
            self.sigma_weight.data.fill_(self.init_s)
            if self.biasp:
                self.mu_bias.data.zero_()
                self.sigma_bias.data.fill_(self.init_s)

    def forward(self, x, sample=True, scale=0.0):
        if self.elementwise_affine:
            with torch.no_grad():
                self.sigma_weight.data = torch.maximum(self.sigma_weight,
                                                       torch.ones_like(self.sigma_weight) * 1e-7).type(
                    self.sigma_weight.dtype).data
                if self.mu_bias is not None:
                    self.sigma_bias.data = torch.maximum(self.sigma_bias, torch.ones_like(self.sigma_bias) * 1e-7).type(
                        self.sigma_bias.dtype).data
        if not sample or not self.elementwise_affine:
            weight = self.mu_weight
            bias = self.mu_bias
        else:
            eps_weight = torch.ones(self.sigma_weight.size(), device=x.device).normal_().type(
                self.sigma_weight.dtype)
            weight = self.mu_weight + self.sigma_weight * eps_weight
            if self.biasp:
                eps_bias = torch.ones(self.sigma_bias.size(), device=x.device).normal_().type(self.sigma_bias.dtype)
                bias = self.mu_bias + self.sigma_bias * eps_bias
            else:
                bias = None

        return F.layer_norm(
            x, self.normalized_shape, weight, bias, self.eps)


if __name__ == '__main__':
    pass
