import torch
import torch.nn as nn

from models.vgg import VGG


vgg_16_model = VGG('VGG16', normalization_layer_name=None)
print(vgg_16_model)
assert isinstance(vgg_16_model.features[1], nn.Identity)