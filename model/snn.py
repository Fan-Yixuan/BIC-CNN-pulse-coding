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


class firing_function(torch.autograd.Function):
    # inference
    @staticmethod
    def forward(ctx, INput):
        ctx.save_for_backward(INput)
        activtions = INput.gt(cfg.MODEL.TH).float()
        return activtions

    # err-propagate
    @staticmethod
    def backward(ctx, grad_output):
        INput, = ctx.saved_tensors
        grad_input = grad_output.clone()
        d_activtions = abs(INput - cfg.MODEL.TH) < cfg.MODEL.LEN
        # d_activtions = torch.exp(-(INput - cfg.MODEL.TH)**2 / (2 * cfg.MODEL.LEN**2)) / ((2 * cfg.MODEL.LEN * np.pi)**0.5)
        return grad_input * d_activtions.float()


act_fun = firing_function.apply


def neuron_block(INput, last_mem, last_spike):
    mem = last_mem * cfg.MODEL.DECAY * (1. - last_spike) + INput
    spike = act_fun(mem)
    return mem, spike


class SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop = nn.Dropout(0.02)
        self.downsample = downsample

    def forward(self, x, c1_mem, c1_spike, c2_mem, c2_spike):
        residual = x
        out = self.conv1(x)
        c1_mem, c1_spike = neuron_block(out, c1_mem, c1_spike)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        c2_mem, c2_spike = neuron_block(out, c2_mem, c2_spike)
        c2_spike = self.drop(c2_spike)
        return c2_spike, c1_mem, c1_spike, c2_mem, c2_spike


class SpikingResNet(nn.Module):
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
        self.fc_custom = nn.Linear(960, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(1. / n))

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * SpikingBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * SpikingBasicBlock.expansion, kernel_size=1, stride=stride, bias=False))

        self.layers.append(SpikingBasicBlock(self.inplanes, planes, stride, downsample).cuda())
        self.inplanes = planes * SpikingBasicBlock.expansion
        for _ in range(1, blocks):
            self.layers.append(SpikingBasicBlock(self.inplanes, planes).cuda())

    def forward(self, *x):
        x = x[0]
        # init variables and create memory for the SNN
        batch_size, time_window, _, w, h = x.size()
        c_mem = c_spike = torch.zeros(batch_size, 64, w // 2, h // 2).cuda()  # //2 due to stride=2
        c2_spike, c2_mem, c1_spike, c1_mem = [], [], [], []
        for i in range(len(self.layer_num)):
            d = self.size_devide[i]
            for _ in range(self.layer_num[i]):
                c1_mem.append(torch.zeros(batch_size, self.planes[i], w // d, h // d).cuda())
                c1_spike.append(torch.zeros(batch_size, self.planes[i], w // d, h // d).cuda())
                c2_mem.append(torch.zeros(batch_size, self.planes[i], w // d, h // d).cuda())
                c2_spike.append(torch.zeros(batch_size, self.planes[i], w // d, h // d).cuda())
        fc_sumspike = fc_mem = fc_spike = torch.zeros(batch_size, self.classes).cuda()

        # SNN window
        for step in range(time_window):
            x_step = x[:, step, :, :, :]
            x_step = self.conv1_custom(x_step)
            c_mem, c_spike = neuron_block(x_step, c_mem, c_spike)
            x_step = self.maxpool1(c_spike)
            for i in range(len(self.layers)):
                x_step, c1_mem[i], c1_spike[i], c2_mem[i], c2_spike[i] = self.layers[i](x_step, c1_mem[i], c1_spike[i], c2_mem[i],
                                                                                        c2_spike[i])
            x_step = torch.cat(c2_spike[0::2], dim=1)
            x_step = self.avgpool2(x_step)
            x_step = x_step.view(x_step.size(0), -1)
            out = self.fc_custom(x_step)
            fc_mem, fc_spike = neuron_block(out, fc_mem, fc_spike)
            fc_sumspike += fc_spike
        fc_sumspike = fc_sumspike / time_window
        return fc_sumspike
