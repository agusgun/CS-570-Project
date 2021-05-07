import torch
import torch.nn as nn

from models.vgg import VGG

def build_network(backbone, normalization_layer_name=None):
    if backbone == 'VGG16':
        return VGG('vgg16', normalization_layer_name=normalization_layer_name)
    elif backbone == 'ResNet50':
        return 
    else:
        raise NotImplementedError()