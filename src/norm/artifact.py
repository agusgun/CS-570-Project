
class AdaptiveGN(nn.Module):
    # Gradient also not flowing here weird solution I think
    def __init__(self, group_num, c_num, examples_group_num = 2, eps = 1e-5, aux_loss=True):
        super(AdaptiveGN, self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.std_param = nn.Parameter(torch.ones(2)) # This is important to enable gradient flow
        self.mean_param = nn.Parameter(torch.ones(1)) # This is important to enable gradient flow

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, C // self.group_num, H, W)
        mean_group_bn = x.mean(dim = [0], keepdim = True)
        std_group_bn = x.std(dim = [0], keepdim = True)

        mean_group_gn = x.mean(dim = [2, 3, 4], keepdim=True)
        std_group_gn = x.std(dim = [2, 3, 4], keepdim=True)
        
        x = std_group_gn * ((x - mean_group_bn) / (std_group_bn + self.eps)) + mean_group_gn
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta

class GBGNNoGroup(nn.Module):
    # Batch Group Norm from Dineen and Summers Simplified (No Grouping in examples)
    def __init__(self, group_num, c_num, examples_group_num = 2, eps = 1e-5, aux_loss=True):
        super(GBGNNoGroup, self).__init__()
        self.group_num = group_num
        self.examples_group_num = examples_group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.std_param = nn.Parameter(torch.ones(2)) # This is important to enable gradient flow
        self.mean_param = nn.Parameter(torch.ones(1)) # This is important to enable gradient flow

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, C // self.group_num, H, W)
        mean = x.mean(dim=[0, 2, 3, 4], keepdim=True)
        std = x.std(dim=[0, 2, 3, 4], keepdim=True)
        x = (x - mean) / (std + self.eps)
        
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta

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

class GroupNormPlusParallel(nn.Module):
    # Old weird version
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

# Follow CBAM way?
class GNPlusSequentialWeighted(nn.Module):
    # Good but less control and weird because the param is in std 
    def __init__(self, group_num, c_num, eps = 1e-5, aux_loss=True):
        super(GNPlusSequentialWeighted, self).__init__()
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
        x = (x - (1 - self.mean_param[0]) * mean) / (self.std_param[1] * std + self.eps)
        
        return x * self.gamma + self.beta


class GNPlusSequential(nn.Module):
    # Good but very small control (Squential Normalization + GN First Without Weight & Updated Sequential Normalization)
    def __init__(self, group_num, c_num, eps = 1e-5, aux_loss=True):
        super(GNPlusSequential, self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        # self.
        self.std_param = nn.Parameter(torch.ones(2))
        self.mean_param = nn.Parameter(torch.ones(1))

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, C // self.group_num, H, W)

        mean = x.mean(dim = [2, 3, 4], keepdim = True)
        std = x.std(dim = [2, 3, 4], keepdim = True)

        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)

        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x = (x - mean) / (std + self.eps)
        
        return x * self.gamma + self.beta


class GBGN(nn.Module):
    # Batch Group Norm from Dineen and Summers (Edge Case works) but this cannot works for small batch size
    def __init__(self, group_num, c_num, examples_group_num = 2, eps = 1e-5, aux_loss=True):
        super(GBGN, self).__init__()
        self.group_num = group_num
        self.examples_group_num = examples_group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps
        self.std_param = nn.Parameter(torch.ones(2)) # This is important to enable gradient flow
        self.mean_param = nn.Parameter(torch.ones(1)) # This is important to enable gradient flow

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(self.examples_group_num, N // self.examples_group_num, self.group_num, C // self.group_num, H, W)
        mean = x.mean(dim=[1, 3, 4, 5], keepdim=True)
        std = x.std(dim=[1, 3, 4, 5], keepdim=True)
        x = (x - mean) / (std + self.eps)
        
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta

class GroupNorm2dExtension(nn.Module):
    # Created to help custom grouping but intractable
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
