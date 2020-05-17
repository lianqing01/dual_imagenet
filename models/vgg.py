'''VGG11/13/16/19 in Pytorch.'''
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from .dual_norm import DualNorm
from .dual_norm import DualAffine
from .constraint_bn_v2 import *
from .batchnorm import BatchNorm2d
from .instancenorm import InstanceNorm2d


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG500': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256] + [256]*500 + ['M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG50': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256] + [256]*50 + ['M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG25': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256] + [256]*25 + ['M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes = 10, with_bn=False):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], with_bn)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, with_bn=False):
        layers = []
        in_channels = 3
        for idx, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if with_bn == 'dual': # deprecated
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               DualNorm(x),
                               nn.ReLU(inplace=True),
                               DualAffine(x),]
                elif with_bn == 'bn':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                elif with_bn == 'bn_population':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               BatchNorm2d(x),
                               nn.ReLU(inplace=True)]

                elif with_bn == 'brn':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               BatchRenorm2d(x, momentum=0.5),
                               nn.ReLU(inplace=True)]

                elif with_bn == 'constraint_bn_v2':
                    if idx == 0:
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [Constraint_Norm2d(in_channels, pre_affine=True, post_affine=True),
                                   #Constraint_Affine2d(in_channels),
                                   nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True)]
                elif with_bn == 'constraint_bn_v3':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                Constraint_Norm2d(x),
                                nn.ReLU(inplace=True)]
                elif with_bn =='gn':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                nn.GroupNorm(64, x, affine=True),
                               nn.ReLU(inplace=True)]
                elif with_bn == 'constraint_bn_v2_no_affine':
                    if idx == 0:
                        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True)]
                    else:
                        layers += [Constraint_Norm2d(in_channels, pre_affine=False, post_affine=False),
                                   #Constraint_Affine2d(in_channels),
                                   nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True)]

                elif with_bn == 'in':
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               InstanceNorm2d(x, affine=True),
                               nn.ReLU(inplace=True),]

                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg16(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn=False)


def vgg500(num_classes=10):
    return VGG('VGG500', num_classes=num_classes, with_bn=False)


def vgg50(num_classes=10):
    return VGG('VGG50', num_classes=num_classes, with_bn=False)
def vgg25(num_classes=10):
    return VGG('VGG25', num_classes=num_classes, with_bn=False)


def vgg500_bn(num_classes=10):
    return VGG('VGG500', num_classes=num_classes, with_bn='bn')
def vgg50_bn(num_classes=10):
    return VGG('VGG50', num_classes=num_classes, with_bn='bn')

def vgg50_constraint_bn_v2(num_classes=10):
    return VGG('VGG50', num_classes=num_classes, with_bn='constraint_bn_v2')

def vgg25_constraint_bn_v2(num_classes=10):
    return VGG('VGG25', num_classes=num_classes, with_bn='constraint_bn_v2')

def vgg16_bn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='bn')
def vgg16_bn_moving_average(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='bn_moving_average')
def vgg16_pn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='bn_population')

def vgg16_in(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='in')

def vgg16_gn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='gn')


def vgg16_brn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='brn')


def vgg16_dual_bn(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='dual')

def vgg16_constraint_bn_v2(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='constraint_bn_v2')

def vgg16_constraint_bn_v3(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='constraint_bn_v3')




def vgg16_constraint_bn_v2_noaffine(num_classes=10):
    return VGG('VGG16', num_classes=num_classes, with_bn='constraint_bn_v2_no_affine')
# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
