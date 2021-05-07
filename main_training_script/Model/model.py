from  torchvision import models
import torch.functional as F
import torch
import torch.nn as nn
import math
from Model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Net_builter(nn.Module):
    def __init__(self,backbone,normalization=None,nbr_class=10):
        super(Net_builter,self).__init__()
        if backbone == 'ResNet50':
            self.model=models.resnet50(pretrained=True)
            self.model.fc=nn.Linear(2048,nbr_class)
        else:
            raise NotImplementedError

        if normalization:
            replace_bn2gn(self.model)


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.model(x)



def replace_bn2gn(module):
    for attr_name in dir(module):
        tar_attr = getattr(module, attr_name)
        if type(tar_attr) == torch.nn.BatchNorm2d:
            print(tar_attr)
            new_bn = torch.nn.GroupNorm(32,tar_attr.num_features, tar_attr.eps, tar_attr.momentum)
            setattr(module, attr_name, new_bn)

    for name, child_module in module.named_children():
        replace_bn2gn(child_module)


if __name__=='__main__':
    model=Net_builter('ResNet50')
    replace_bn2gn(model)
    print(model)

