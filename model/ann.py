import logging
import os
import sys

import numpy as np
import torch
from torch import nn

root = os.getcwd()
sys.path.append(root)
from config import cfg

logger = logging.getLogger()


class Encoder(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.encoder = nn.Conv2d(1, window, kernel_size=1, bias=False)
        if cfg.TRAIN.ONE_INIT:
            init_weight = torch.Tensor([[[[1]]]] * window)
            self.encoder.weight = torch.nn.Parameter(init_weight)

    def forward(self, *x):
        x = x[0]
        return self.encoder(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop = nn.Dropout(0.02)
        self.downsample = downsample

    def forward(self, *x):
        x = x[0]
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.drop(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layer_num, classes, channel):
        super().__init__()
        self.inplanes = 64
        self.classes = classes

        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        self.layer_num = layer_num
        self.size_devide = np.array([4, 4, 4, 4])
        self.planes = [64, 128, 256, 512]
        self._make_layer(self.planes[0], layer_num[0], stride=1)
        self._make_layer(self.planes[1], layer_num[1], stride=1)
        self._make_layer(self.planes[2], layer_num[2], stride=1)
        self._make_layer(self.planes[3], layer_num[3], stride=1)

        self.avgpool2 = nn.AvgPool2d(kernel_size=7)
        self.fc_custom = nn.Linear(512, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(1. / n))

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False))

        self.layers.append(BasicBlock(self.inplanes, planes, stride, downsample).cuda())
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            self.layers.append(BasicBlock(self.inplanes, planes).cuda())

    def forward(self, *x):
        x = x[0]
        x = self.conv1_custom(x)
        x = self.maxpool1(x)
        for cnt in range(len(self.layers)):
            x = self.layers[cnt](x)
        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)
        out = self.fc_custom(x)
        return out
