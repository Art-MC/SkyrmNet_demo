"""
A fully convolutional U-net like neural network for image segmentation.  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 

class smallUnet(nn.Module):
    '''
    Builds a fully convolutional Unet-like neural network model
    Args:
        nb_classes: int
            number of classes in the ground truth
        nb_filters: int
            number of filters in 1st convolutional block
            (gets multibplied by 2 in each next block)
        use_dropout: bool
            use / not use dropout in the 3 inner layers
        batch_norm: bool
            use / not use batch normalization after each convolutional layer
        upsampling mode: str
            "bilinear" or "nearest" upsampling method.
            Bilinear is usually more accurate, but adds additional (small) 
            randomness; for full reproducibility, consider using 'nearest'
            (this assumes that all other sources of randomness are fixed)
    '''
    def __init__(self,
                 nb_classes=2,
                 nb_filters=16,
                 use_dropout=False,
                 batch_norm=True,
                 upsampling_mode="nearest"):
        super(smallUnet, self).__init__()
        dropout_vals = [.1, .2, .1] if use_dropout else [0, 0, 0]
        self.c1 = conv2dblock(
            1, 1, nb_filters,
            use_batchnorm=batch_norm)

        self.c2 = conv2dblock(
            2, nb_filters, nb_filters*2,
            use_batchnorm=batch_norm)

        self.c3 = conv2dblock(
            2, nb_filters*2, nb_filters*4,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[0])

        self.bn = conv2dblock(
            3, nb_filters*4, nb_filters*8,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[1])

        self.upsample_block1 = upsample_block(
            nb_filters*8, nb_filters*4,
            mode=upsampling_mode)

        self.c4 = conv2dblock(
            2, nb_filters*8, nb_filters*4,
            use_batchnorm=batch_norm,
            dropout_=dropout_vals[2])

        self.upsample_block2 = upsample_block(
            nb_filters*4, nb_filters*2,
            mode=upsampling_mode)

        self.c5 = conv2dblock(
            2, nb_filters*4, nb_filters*2,
            use_batchnorm=batch_norm)

        self.upsample_block3 = upsample_block(
            nb_filters*2, nb_filters,
            mode=upsampling_mode)

        self.c6 = conv2dblock(
            1, nb_filters*2, nb_filters,
            use_batchnorm=batch_norm)
        
        self.px = nn.Conv2d(nb_filters, nb_classes, 1, 1, 0)
        self.maxpool = F.max_pool2d
        self.concat = torch.cat

    def forward(self, x):
        '''Defines a forward path'''
        # Contracting path
        c1 = self.c1(x)
        d1 = self.maxpool(c1, kernel_size=2, stride=2)
        c2 = self.c2(d1)
        d2 = self.maxpool(c2, kernel_size=2, stride=2)
        c3 = self.c3(d2)
        d3 = self.maxpool(c3, kernel_size=2, stride=2)
        # Bottleneck layer
        bn = self.bn(d3)
        # Expanding path
        u3 = self.upsample_block1(bn)
        u3 = self.concat([c3, u3], dim=1)
        u3 = self.c4(u3)
        u2 = self.upsample_block2(u3)
        u2 = self.concat([c2, u2], dim=1)
        u2 = self.c5(u2)
        u1 = self.upsample_block3(u2)
        u1 = self.concat([c1, u1], dim=1)
        u1 = self.c6(u1)
        # Final layer used for pixel-wise convolution
        px = self.px(u1)
        return px



class conv2dblock(nn.Module):
    '''
    Creates block(s) consisting of convolutional
    layer, leaky relu and (optionally) dropout and
    batch normalization
    '''
    def __init__(self, nb_layers, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=1, use_batchnorm=False,
                 lrelu_a=0.01, dropout_=0):
        '''Initializes module parameters'''
        super(conv2dblock, self).__init__()
        block = []
        for idx in range(nb_layers):
            input_channels = output_channels if idx > 0 else input_channels
            block.append(nn.Conv2d(input_channels,
                                   output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding))
            if dropout_ > 0:
                block.append(nn.Dropout(dropout_))
            block.append(nn.LeakyReLU(negative_slope=lrelu_a))
            if use_batchnorm:
                block.append(nn.BatchNorm2d(output_channels))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        '''Forward path'''
        output = self.block(x)
        return output


class upsample_block(nn.Module):
    '''
    Defines upsampling block. The upsampling is performed 
    using bilinear or nearest interpolation followed by 1-by-1
    convolution (the latter can be used to reduce
    a number of feature channels).
    '''
    def __init__(self,
                 input_channels,
                 output_channels,
                 scale_factor=2,
                 mode="bilinear"):
        '''Initializes module parameters'''
        super(upsample_block, self).__init__()
        assert mode == 'bilinear' or mode == 'nearest',\
            "use 'bilinear' or 'nearest' for upsampling mode"
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''Defines a forward path'''
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


def rng_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False