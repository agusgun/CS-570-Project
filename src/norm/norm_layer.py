import torch
import torch.nn as nn
import numpy as np

# Original Implementation of GN
class GroupNorm2d(nn.Module):
    def __init__(self, group_num, c_num, eps = 1e-5):
        super(GroupNorm2d,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim = 2, keepdim = True)
        std = x.std(dim = 2, keepdim = True)
        
        x = (x - mean) / (std+self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta

# Our Proposed Solution
class GNPlusParallel(nn.Module):
    def __init__(self, group_num, c_num, eps = 1e-5):
        super(GNPlusParallel,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.lambda_param = nn.Parameter(torch.ones(1))

    def forward(self, x):
        N, C, H, W = x.size()

        x_gn = x.view(N, self.group_num, C // self.group_num, H, W)
        mean_gn = x_gn.mean(dim = [2, 3, 4], keepdim = True)
        std_gn = x_gn.std(dim = [2, 3, 4], keepdim = True)
        x_gn = (x_gn - mean_gn) / (std_gn + self.eps)
        x_gn = x_gn.view(N, C, H, W)

        mean_bn = x.mean(dim=0, keepdim=True)
        std_bn = x.std(dim=0, keepdim=True)
        x = (x -  mean_bn) / (std_bn + self.eps)
        
        return (torch.sigmoid(self.lambda_param[0]) * x_gn + (1 - torch.sigmoid(self.lambda_param[0])) * x) * self.gamma + self.beta

# Our Proposed Solution
class GNPlusSequentialGNFirst(nn.Module):
    def __init__(self, group_num, c_num, eps = 1e-5):
        super(GNPlusSequentialGNFirst,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.lambda_param = nn.Parameter(torch.ones(1))

    def forward(self, x):
        N, C, H, W = x.size()

        x_gn = x.view(N, self.group_num, C // self.group_num, H, W)
        mean_gn = x_gn.mean(dim = [2, 3, 4], keepdim = True)
        std_gn = x_gn.std(dim = [2, 3, 4], keepdim = True)
        x_gn = (x_gn - mean_gn) / (std_gn + self.eps)
        x_gn = x_gn.view(N, C, H, W)

        mean_bn = x_gn.mean(dim=0, keepdim=True)
        std_bn = x_gn.std(dim=0, keepdim=True)
        x_bn = (x_gn -  mean_bn) / (std_bn + self.eps)
        
        return (torch.sigmoid(self.lambda_param[0]) * x_gn + (1 - torch.sigmoid(self.lambda_param[0])) * x_bn) * self.gamma + self.beta

# Our Proposed Solution
class GNPLusSequentialBNFirst(nn.Module):
    def __init__(self, group_num, c_num, eps = 1e-5):
        super(GNPLusSequentialBNFirst,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.lambda_param = nn.Parameter(torch.ones(1))

    def forward(self, x):
        N, C, H, W = x.size()

        mean_bn = x.mean(dim=0, keepdim=True)
        std_bn = x.std(dim=0, keepdim=True)
        x_bn = (x - mean_bn) / (std_bn + self.eps)

        x_gn = x_bn.view(N, self.group_num, C // self.group_num, H, W)
        mean_gn = x_gn.mean(dim = [2, 3, 4], keepdim = True)
        std_gn = x_gn.std(dim = [2, 3, 4], keepdim = True)
        x_gn = (x_gn - mean_gn) / (std_gn + self.eps)
        x_gn = x_gn.view(N, C, H, W)

        return (torch.sigmoid(self.lambda_param[0]) * x_gn + (1 - torch.sigmoid(self.lambda_param[0])) * x_bn) * self.gamma + self.beta

def get_norm_layer(c_out, n_group=32, norm='bn'):
    if norm == 'bn' or norm == 'bn_regularization':
        return nn.BatchNorm2d(c_out)
    elif norm == 'in':
        return nn.InstanceNorm2d(c_out)
    elif norm == 'gn' or norm == 'gn_regularization':
        return GroupNorm2d(group_num=n_group, c_num=c_out)
    elif norm =='gn_noisy':
        return GroupNorm2d(group_num=n_group, c_num=c_out)
    elif norm == 'gn_plus_gn_first_noisy':
        return GNPlusSequentialGNFirst(group_num=n_group, c_num=c_out)
    elif norm == 'gn_plus_sequential_gn_first':
        return GNPlusSequentialGNFirst(group_num=n_group, c_num=c_out)
    elif norm == 'gn_plus_sequential_bn_first':
        return GNPlusSequentialBNFirst(group_num=n_group, c_num=c_out)
    elif norm == 'gn_plus_parallel':
        return GNPlusParallel(group_num=n_group, c_num=c_out)
    elif norm == 'bn_noisy':
        return nn.BatchNorm2d(c_out)
    elif norm == None or norm == 'regularization':
        return nn.Identity()

def replace_norm_layer(layer, new_norm_layer='gn'):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    new_layer = new_norm_layer
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    if new_norm_layer == 'gn':
                        layer._modules[name] = nn.GroupNorm(32, num_channels)
                    elif new_norm_layer == 'gn_noisy':
                        layer._modules[name] = GroupNormNoisy2d(group_num=32, c_num=num_channels)
                    elif new_norm_layer == None:
                        layer._modules[name] = nn.Identity()
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = replace_norm_layer(sub_layer, new_norm_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer

if __name__ == "__main__":
    x = torch.randn(8, 64, 3, 3, requires_grad=True)
    GN = GroupNorm2d(4, 64)
    GN_Extension = GroupNorm2dExtension(4, 64)
    output_GN = GN(x)
    
    criterion = nn.MSELoss()
    true_label = torch.ones(8, 64, 3, 3)
    loss = criterion(output_GN, true_label)
    loss.backward()
    x_grad_first = x.grad
    
    output_GN_Extension = GN_Extension(x)
    loss = criterion(output_GN_Extension, true_label)
    loss.backward()
    x_grad_second = x.grad
    print(torch.all(torch.eq(x_grad_first, x_grad_second)))
    print(torch.all(torch.eq(output_GN, output_GN_Extension)))

    GN = GroupNorm2dPlus(4, 64)
    GN(x)