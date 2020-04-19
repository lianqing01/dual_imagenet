import torch

import torch.nn as nn
from .constraint_bn_v2 import *


class Two_layer(nn.Module):
    def __init__(self, input_dim=3072, num_classes=10):
        super(Two_layer, self).__init__()
        self.layer1 = nn.Linear(input_dim, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.constraint_bn = Constraint_Norm1d(num_classes)
        self.constraint_affine = Constraint_Affine1d(num_classes)

    def forward(self, x):
        bsz = x.size(0)
        x = x.view(bsz, -1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.constraint_bn(x)
        x = self.constraint_affine(x)
        return x






