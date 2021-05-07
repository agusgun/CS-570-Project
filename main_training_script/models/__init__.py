import torch
import torch.nn as nn

from models.vgg import VGG
from models.resnet import ResNet50

def build_network(net_name, num_classes, normalization_layer_name=None):
    if net_name == 'VGG16':
        return VGG('VGG16', num_classes=num_classes, normalization_layer_name=normalization_layer_name)
    elif net_name == 'ResNet50':
        return ResNet50(num_classes=num_classes, normalization_layer_name=normalization_layer_name)
    else:
        raise NotImplementedError()