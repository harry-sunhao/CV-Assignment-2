import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def deconv_layer(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv = conv_layer(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               batch_norm=True)

    def forward(self, x):
        return x + self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        self.opts = opts

        #####TODO: Define the discriminator network#####
        pass
        ################################################

    def forward(self, x):
        #####TODO: Define the forward pass#####
        pass
        #######################################


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.opts = opts

        #####TODO: Define the generator network######
        pass
        #############################################

    def forward(self, x):
        #####TODO: Define the forward pass#####
        pass
        #######################################


class CycleGenerator(nn.Module):
    def __init__(self, opts):
        super(CycleGenerator, self).__init__()
        self.opts = opts

        #####TODO: Define the cyclegan generator network######
        pass
        ######################################################

    def forward(self, x):
        #####TODO: Define the forward pass#####
        pass
        #######################################
