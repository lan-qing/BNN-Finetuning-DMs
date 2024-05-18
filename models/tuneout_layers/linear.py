import math
import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class TuneoutLinear(nn.Module):
    def __init__(self, init_module, rate):
        super(TuneoutLinear, self).__init__()
        self.delta_module = nn.Linear(in_features=init_module.in_features, out_features=init_module.out_features,
                                      bias=init_module.bias, device=init_module.device, dtype=init_module.dtype)
        self.delta_module.weight.data.zero_(0)
        self.delta_module.bias.data.zero_(0)
        self.init_module = init_module
        self.init_module.requires_grad = False
        self.rate = rate

    def forward(self, x):
        delta_output = F.dropout(self.delta_module(x), p=self.rate, training=self.training)
        with torch.no_grad():
            init_output = self.init_module(x)
        return delta_output + init_output
