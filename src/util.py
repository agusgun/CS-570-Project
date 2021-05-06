import torch
import torch.nn as nn

import inspect

def conv(c_in, c_out, k_size, stride=2, pad=1, bias=False, norm='bn', activation=None):
    layers = []

    # Conv.
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=bias))

    # Normalization
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(c_out))
    elif norm == 'in':
        layers.append(nn.InstanceNorm2d(c_out))
    elif inspect.isclass(norm): # custom normalization
        layers.appened(norm)
    elif norm == None:
        pass

    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif activation == None:
        pass

    return nn.Sequential(*layers)
