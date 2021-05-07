import torch.nn as nn

from Model import model


def build_backbone(backbone, BatchNorm=nn.BatchNorm2d):
    if backbone == 'ResNet101':
        return model.ResNet101(BatchNorm)
    else:
        raise NotImplementedError