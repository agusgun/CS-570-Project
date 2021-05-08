# Moved from Experimentation to Testing
import torch
import torch.nn as nn

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
        self.lambda_param = nn.Parameter(torch.tensor([0.5]))

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
        
        return (self.lambda_param[0] * x_gn + (1 - self.lambda_param[0]) * x) * self.gamma + self.beta

# Our Proposed Solution
class GNPlusSequentialGNFirst(nn.Module):
    def __init__(self, group_num, c_num, eps = 1e-5):
        super(GNPlusSequentialGNFirst,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.lambda_param = nn.Parameter(torch.tensor([0.5]))

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
        
        return (self.lambda_param[0] * x_gn + (1 - self.lambda_param[0]) * x_bn) * self.gamma + self.beta

# Our Proposed Solution
class GNPLusSequentialBNFirst(nn.Module):
    def __init__(self, group_num, c_num, eps = 1e-5):
        super(GNPLusSequentialBNFirst,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.lambda_param = nn.Parameter(torch.tensor([0.5]))

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

        return (self.lambda_param[0] * x_gn + (1 - self.lambda_param[0]) * x_bn) * self.gamma + self.beta

def get_norm_layer(c_out, n_group=32, norm='bn'):
    if norm == 'bn':
        return nn.BatchNorm2d(c_out)
    elif norm == 'gn':
        return GroupNorm2d(group_num=n_group, c_num=c_out)
    elif norm == 'gn_plus_sequential_gn_first':
        return GNPlusSequentialGNFirst(group_num=n_group, c_num=c_out)
    elif norm == 'gn_plus_sequential_bn_first':
        return GNPLusSequentialBNFirst(group_num=n_group, c_num=c_out)
    elif norm == 'gn_plus_parallel':
        return GNPlusParallel(group_num=n_group, c_num=c_out)
    else:
        return nn.Identity()