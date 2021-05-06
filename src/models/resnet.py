import torch
import torch.nn as nn
import torch.nn.functional as F

from src.util import conv
from src.norm.norm_layer import get_norm_layer


# Reference: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, norm='bn', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = get_norm_layer(c_out=out_channels, norm=norm)
        print(self.norm1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm2 = get_norm_layer(c_out=out_channels, norm=norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                get_norm_layer(c_out=self.expansion * out_channels, norm=norm)
            )
    
    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, norm='bn', stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = get_norm_layer(c_out=out_channels, norm=norm)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.norm2 = get_norm_layer(c_out=out_channels, norm=norm)
        self.conv3 = nn.Conv2d(out_channels, self.expansion *
                               out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(self.expansion*out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                get_norm_layer(c_out=self.expansion * out_channels, norm=norm)
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, norm='bn', num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm1 = get_norm_layer(c_out=64, norm=norm)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm=norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm=norm)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, norm, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet50(norm='bn', num_classes=1000):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], norm=norm, num_classes=num_classes)
