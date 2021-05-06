import torch
import torch.nn as nn
import numpy as np

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

class GroupNorm2dExtension(nn.Module):
    def __init__(self, group_num, c_num, eps = 1e-5):
        super(GroupNorm2dExtension,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        
        # Group Label for each Channel
        self.group_label = np.zeros(c_num, dtype=np.int32)
        num_each_group = c_num // self.group_num
        self.counter_for_each_label = {}
        for i in range(self.group_num):
            self.group_label[i * num_each_group:i * num_each_group + num_each_group] = i
            self.counter_for_each_label[i] = len(self.group_label[i * num_each_group: i * num_each_group + num_each_group])

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, C, -1)

        # Mean and Std for each group
        new_counter_for_each_label = np.zeros(self.group_num, dtype=np.int32) # for counting later
        activations_for_each_label = {}
        for label, count in  self.counter_for_each_label.items():
            new_counter_for_each_label[label] = 0
            activations_for_each_label[label] = torch.zeros(N, count, H * W, device=x.device)

        counter = 0
        for label in self.group_label:
            activations_for_each_label[label][:, new_counter_for_each_label[label], :] = x[:, counter, :]
            counter += 1
            new_counter_for_each_label[label] += 1
        
        mean_for_each_label = {}
        std_for_each_label = {}
        for label, activations in activations_for_each_label.items():
            activations = activations.view(N, -1)
            mean_for_each_label[label] = activations.mean(dim=1)
            std_for_each_label[label] = activations.std(dim=1)
        
        # Normalize each channel
        y = x.clone()
        for i in range(C):
            mean = mean_for_each_label[self.group_label[i]].view(N, 1)
            std = std_for_each_label[self.group_label[i]].view(N, 1)
            y[:, i, :] = (y[:, i, :].clone() - mean) / (std + self.eps)
        x = y.view(N, C, H, W)

        return x * self.gamma + self.beta

# Follow CBAM way?
class GroupNormPlusSequential(nn.Module):
    def __init__(self, group_num, c_num, eps = 1e-5, aux_loss=True):
        super(GroupNormPlusSequential, self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        # self.
        self.std_param = nn.Parameter(torch.ones(2))
        self.mean_param = nn.Parameter(torch.ones(1))

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim = 2, keepdim = True)
        std = x.std(dim = 2, keepdim = True)

        x = (x - self.mean_param[0] * mean) / (self.std_param[0] * std + self.eps)
        x = x.view(N, C, H, W)

        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x = (x - (1 - self.mean_param[0]) * mean) / ((self.std_param[1]) * std + self.eps)
        
        return x * self.gamma + self.beta

class BGN(nn.Module):
    # Gradient also not flowing here
    def __init__(self, group_num, c_num, examples_group_num = 2, eps = 1e-5, aux_loss=True):
        super(BGN, self).__init__()
        self.group_num = group_num
        self.examples_group_num = examples_group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.std_param = nn.Parameter(torch.ones(2)) # This is important to enable gradient flow
        self.mean_param = nn.Parameter(torch.ones(1)) # This is important to enable gradient flow

    def forward(self, x):
        N, C, H, W = x.size()

        # x = x.view(self.examples_group_num, N // self.examples_group_num, self.group_num, C // self.group_num, H, W)
        # BatchGroupNorm from Dineen and Summers also doesn't works
        # This technique also doesn't work
        x = x.view(N, self.group_num, C // self.group_num, H, W)
        mean_group_bn = x.mean(dim = [0], keepdim = True)
        std_group_bn = x.std(dim = [0], keepdim = True)

        mean_group_gn = x.mean(dim = [2, 3, 4], keepdim=True)
        std_group_gn = x.std(dim = [2, 3, 4], keepdim=True)

        x = (x - (self.mean_param[0] * mean_group_gn)) / (self.std_param[0] * std_group_gn + self.eps)
        x = (x - ((1 - self.mean_param[0]) * mean_group_bn)) / (self.std_param[1] * std_group_bn + self.eps)
        
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta

# GN BN
class GNBN(nn.Module):
    # Gradient cannot flow 0.1 LR + Noisy, 0.01 LR also cannot flow
    def __init__(self, group_num, c_num, eps = 1e-5, aux_loss=True):
        super(GNBN, self).__init__()
        self.group_num = group_num
        self.eps = eps
        self.gn = GroupNormPlusParallel(group_num=group_num, c_num=c_num)
        self.bn = nn.BatchNorm2d(c_num)

    def forward(self, x):
        x = self.gn(x)
        x = self.bn(x)
        return x

class GroupNormBNFirst(nn.Module):
    # Unflowing gradient also occurs in this improvement we need to undo the effect of GN
    def __init__(self, group_num, c_num, eps = 1e-5, aux_loss=True):
        super(GroupNormBNFirst,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.lambda_param = nn.Parameter(torch.ones(1))

    def forward(self, x):
        N, C, H, W = x.size()

        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x = (x - (1 - self.lambda_param[0]) * mean) / ((1 - self.lambda_param[0]) * std + self.eps)
        
        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim = 2, keepdim = True)
        std = x.std(dim = 2, keepdim = True)

        x = (x - self.lambda_param[0] * mean) / (self.lambda_param[0] * std + self.eps)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta

class GroupNormPlusParallel(nn.Module):
    # Unflowing gradient also occurs here
    def __init__(self, group_num, c_num, eps = 1e-5, aux_loss=True):
        super(GroupNormPlusParallel,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.lambda_param = nn.Parameter(torch.ones(1))

    def forward(self, x):
        N, C, H, W = x.size()

        x_gn = x.view(N, self.group_num, -1)
        mean_gn = x_gn.mean(dim = 2, keepdim = True)
        std_gn = x_gn.std(dim = 2, keepdim = True)
        x_gn = (x_gn - self.lambda_param[0] * mean_gn) / (self.lambda_param[0] * std_gn + self.eps)
        x_gn = x_gn.view(N, C, H, W)

        mean_bn = x.mean(dim=0, keepdim=True)
        std_bn = x.std(dim=0, keepdim=True)
        x = (x - (1 - self.lambda_param[0]) * mean_bn) / ((1 - self.lambda_param[0]) * std_bn + self.eps)
        
        return (x + x_gn) * self.gamma + self.beta

def get_norm_layer(c_out, n_group=32, norm='bn'):
    if norm == 'bn' or norm == 'bn_regularization':
        return nn.BatchNorm2d(c_out)
    elif norm == 'in':
        return nn.InstanceNorm2d(c_out)
    elif norm == 'gn' or norm == 'gn_regularization':
        return GroupNorm2d(group_num=n_group, c_num=c_out)
    elif norm =='gn_noisy':
        # return GroupNormPlusSequential(group_num=n_group, c_num=c_out)
        return BGN(group_num=n_group, c_num=c_out)
        # return GNBN(group_num=n_group, c_num=c_out)
    elif norm == 'gn_plus_sequential':
        return BGN(group_num=n_group, c_num=c_out)
    elif norm == 'gn_plus_parallel':
        return GroupNormPlusParallel(group_num=n_group, c_num=c_out)
    elif norm == 'gn_extension':
        return GroupNorm2dExtension(group_num=n_group, c_num=c_out)
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