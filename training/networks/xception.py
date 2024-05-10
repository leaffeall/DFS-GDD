'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706

The code is mainly modified from GitHub link below:
https://github.com/ondyari/FaceForensics/blob/master/classification/network/xception.py
'''

import os
import argparse
import logging

import math

import numpy as np
import torch
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo
from torch.nn import init
from typing import Union
from utils.registry import BACKBONE

logger = logging.getLogger(__name__)



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self,x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class Block_part(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block_part, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(Partial_conv3(filters))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class Block_part1(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block_part1, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(Partial_conv3(filters))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(Partial_conv3(filters))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

def add_gaussian_noise(ins, mean=0, stddev=0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise


@BACKBONE.register_module(module_name="xception")
class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, xception_config):
        """ Constructor
        Args:
            xception_config: configuration file with the dict format
        """
        super(Xception, self).__init__()
        self.num_classes = xception_config["num_classes"]
        self.mode = xception_config["mode"]
        inc = xception_config["inc"]
        dropout = xception_config["dropout"]

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # Entry flowssssssssssssss
        self.conv1s = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)

        self.bn1s = nn.BatchNorm2d(32)
        self.relus = nn.ReLU(inplace=True)

        self.conv2s = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2s = nn.BatchNorm2d(64)
        # do relu here

        self.block1s = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2s = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3s = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4s = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5s = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6s = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7s = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8s = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9s = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10s = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11s = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12s = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3s = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3s = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4s = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4s = nn.BatchNorm2d(2048)

        # self.last_linear = nn.Linear(2048, self.num_classes)
        # if dropout:
        #     self.last_linear = nn.Sequential(
        #         nn.Dropout(p=dropout),
        #         nn.Linear(2048, self.num_classes)
        #     )
        #
        # self.adjust_channel = nn.Sequential(
        #     nn.Conv2d(2048, 512, 1, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # )


        # self.pag1 = PagFM(728, 364)
        # self.pag2 = PagFM(728, 364)
        # self.pag3 = PagFM(2048, 1024)
        # self.get_high1=GaussianFilter(3, 1.0,in_channels=32)
        # self.get_high2=GaussianFilter(3, 1.0,in_channels=64)


           
    # def fea_part1_0(self, x,y):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #
    #     y = self.conv1s(y)
    #     y = self.bn1s(y)
    #     y = self.relus(y)
    #
    #     return x,y
    #
    # def fea_part1_1(self, x,y):
    #
    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = self.relu(x)
    #
    #     y = self.conv2s(y)
    #     y = self.bn2s(y)
    #     y = self.relus(y)
    #
    #     return x,y
    
    def fea_part1(self, x,y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        y = self.conv1s(y)
        y = self.bn1s(y)
        y = self.relus(y)


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        y = self.conv2s(y)
        y = self.bn2s(y)
        y = self.relus(y)



        return x,y
    
    def fea_part2(self, x,y):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        y = self.block1s(y)
        y = self.block2s(y)
        y = self.block3s(y)


        return x,y

    def fea_part3(self, x,y):

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        y = self.block4s(y)
        y = self.block5s(y)
        y = self.block6s(y)
        y = self.block7s(y)


        return x,y

    def fea_part4(self, x,y):
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        y = self.block8s(y)
        y = self.block9s(y)
        y = self.block10s(y)
        y = self.block11s(y)


        x = self.block12(x)
        y = self.block12s(y)
        return x,y

    def fea_part5(self, x,y):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        y = self.conv3s(y)
        y = self.bn3s(y)
        y = self.relus(y)

        y = self.conv4s(y)
        y = self.bn4s(y)

        return x,y
     
    def features(self, x,y):
        x,y = self.fea_part1(x,y)
        x,y = self.fea_part2(x,y)
        x,y = self.fea_part3(x,y)
        x,y = self.fea_part4(x,y)

        x,y = self.fea_part5(x,y)

        x = x + y

        
        return x

    def classifier(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x



class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y

        return x


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size, sigma, in_channels=3):
        super(GaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.in_channels = in_channels
        self.kernel = nn.Parameter(self.create_high_pass_kernel())

    def create_high_pass_kernel(self):
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        pad = self.kernel_size // 2
        for x in range(-pad, -pad + self.kernel_size):
            for y in range(-pad, -pad + self.kernel_size):
                kernel[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (self.sigma ** 2)))
        kernel /= (self.sigma ** 2 * np.pi * 2)
        kernel /= kernel.sum()
        identity_kernel = np.zeros((self.kernel_size, self.kernel_size))
        identity_kernel[pad, pad] = 1
        high_pass_kernel = identity_kernel - kernel
        high_pass_kernel /= -high_pass_kernel[pad, pad]
        return torch.FloatTensor(high_pass_kernel).unsqueeze(0).unsqueeze(0).repeat(self.in_channels, 1, 1, 1)

    def reset_center(self):
        with torch.no_grad():
            pad = self.kernel_size // 2
            center_values = self.kernel.data[:, :, pad, pad].clone()
            center_values.unsqueeze_(-1).unsqueeze_(-1)
            sum_except_center = torch.sum(self.kernel.data, dim=(2, 3), keepdim=True) - center_values
            self.kernel.data /= sum_except_center
            self.kernel.data[:, :, pad, pad] = -1

    def forward(self, x):
        device = x.device
        return F.conv2d(x, self.kernel.to(device), padding=self.kernel_size // 2, groups=self.in_channels)
