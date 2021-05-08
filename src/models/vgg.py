import torch.nn as nn
import torch
from src.norm.norm_layer import get_norm_layer

class GaussianNoise(nn.Module):
    def __init__(self, group_num, mean, stddev):
        super().__init__()
        self.mean = mean
        self.stddev = stddev
        self.group_num = group_num

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        if self.training:
            x = x + torch.normal(self.mean, self.stddev, size=(N, self.group_num, 1), device=x.device, requires_grad=True)
        x = x.view(N, C, H, W)

        return x

class VGG16(nn.Module):
    # VGG A not 16, sorry for wrong naming
    def __init__(self, inp_ch=3, num_classes=10, norm=None, mean=0.5, std=1.25):
        super().__init__()
        mean = mean
        std = std

        stage1 = []
        stage1.append(nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1))
        stage1.append(get_norm_layer(c_out=64, norm=norm))
        if norm == 'gn_noisy' or norm == 'bn_noisy':
            stage1.append(GaussianNoise(32, mean, std))
        stage1.append(nn.ReLU())
        stage1.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if norm == 'gn_regularization' or norm == 'bn_regularization' or norm == 'regularization':
            stage1.append(nn.Dropout(0.3))
        self.stage1 = nn.Sequential(*stage1)
        
        stage2 = []
        stage2.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
        stage2.append(get_norm_layer(c_out=128, norm=norm))
        if norm == 'gn_noisy' or norm == 'bn_noisy':
            stage2.append(GaussianNoise(32, mean, std))
        stage2.append(nn.ReLU())
        stage2.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if norm == 'gn_regularization' or norm == 'bn_regularization' or norm == 'regularization':
            stage2.append(nn.Dropout(0.3))
        self.stage2 = nn.Sequential(*stage2)
        
        stage3 = []
        stage3.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
        stage3.append(get_norm_layer(c_out=256, norm=norm))
        if norm == 'gn_noisy' or norm == 'bn_noisy':
            stage3.append(GaussianNoise(32, mean, std))
        stage3.append(nn.ReLU())
        stage3.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        stage3.append(get_norm_layer(c_out=256, norm=norm))
        if norm == 'gn_noisy' or norm == 'bn_noisy':
            stage3.append(GaussianNoise(32, mean, std))
        stage3.append(nn.ReLU())
        stage3.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if norm == 'gn_regularization' or norm == 'bn_regularization' or norm == 'regularization':
            stage3.append(nn.Dropout(0.3))
        self.stage3 = nn.Sequential(*stage3)
        
        stage4 = []
        stage4.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1))
        stage4.append(get_norm_layer(c_out=512, norm=norm))
        if norm == 'gn_noisy' or norm == 'bn_noisy':
            stage4.append(GaussianNoise(32, mean, std))
        stage4.append(nn.ReLU())
        stage4.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        stage4.append(get_norm_layer(c_out=512, norm=norm))
        if norm == 'gn_noisy' or norm == 'bn_noisy':
            stage4.append(GaussianNoise(32, mean, std))
        stage4.append(nn.ReLU())
        stage4.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if norm == 'gn_regularization' or norm == 'bn_regularization' or norm == 'regularization':
            stage4.append(nn.Dropout(0.3))
        self.stage4 = nn.Sequential(*stage4)
        
        stage5 = []
        stage5.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        stage5.append(get_norm_layer(c_out=512, norm=norm))
        if norm == 'gn_noisy' or norm == 'bn_noisy':
            stage5.append(GaussianNoise(32, mean, std))
        stage5.append(nn.ReLU())
        stage5.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
        stage5.append(get_norm_layer(c_out=512, norm=norm))
        if norm == 'gn_noisy' or norm == 'bn_noisy':
            stage5.append(GaussianNoise(32, mean, std))
        stage5.append(nn.ReLU())
        stage5.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if norm == 'gn_regularization' or norm == 'bn_regularization' or norm == 'regularization':
            stage5.append(nn.Dropout(0.3))
        self.stage5 = nn.Sequential(*stage5)
        
        stage6 = []
        stage6.append(nn.Linear(512*1*1, 512))
        stage6.append(nn.ReLU())
        stage6.append(nn.Linear(512, 512))
        stage6.append(nn.ReLU())
        stage6.append(nn.Linear(512, num_classes))
        self.stage6 = nn.Sequential(*stage6)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x.view(-1, 512*1*1))
        return x
