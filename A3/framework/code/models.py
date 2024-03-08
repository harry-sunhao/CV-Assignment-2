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
        self.discriminator_channels = opts.discriminator_channels
        # self.discriminator_channels = [32, 64, 128, 1]
        self.conv1 = conv_layer(in_channels=3, out_channels=self.opts.discriminator_channels[0], kernel_size=4)
        self.conv2 = conv_layer(in_channels=self.opts.discriminator_channels[0],
                                out_channels=self.opts.discriminator_channels[1],
                                kernel_size=4)
        self.conv3 = conv_layer(in_channels=self.opts.discriminator_channels[1],
                                out_channels=self.opts.discriminator_channels[2],
                                kernel_size=4)
        self.conv4 = conv_layer(in_channels=self.opts.discriminator_channels[2],
                                out_channels=self.opts.discriminator_channels[3],
                                kernel_size=4, stride=1, padding=0, batch_norm=False)

        # TODO: Define the discriminator network
        # pass
        ################################################

    def forward(self, x):
        # TODO: Define the forward pass#####
        out = F.relu(self.conv1(x))  # BS x 64 x 16 x 16
        out = F.relu(self.conv2(out))  # BS x 64 x 8 x 8
        out = F.relu(self.conv3(out))  # BS x 64 x 4 x 4

        out = self.conv4(out).squeeze()

        return out


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.opts = opts
        # opts.noise_size = 100
        # opts.generator_channels = [128, 64, 32, 3]
        self.linear_bn = deconv_layer(in_channels=opts.noise_size, out_channels=opts.generator_channels[0],
                                      kernel_size=4, stride=2, padding=0)
        self.deconv1 = deconv_layer(in_channels=opts.generator_channels[0], out_channels=opts.generator_channels[1],
                                    kernel_size=4)
        self.deconv2 = deconv_layer(in_channels=opts.generator_channels[1], out_channels=opts.generator_channels[2],
                                    kernel_size=4)
        self.deconv3 = deconv_layer(in_channels=opts.generator_channels[2], out_channels=opts.generator_channels[3],
                                    kernel_size=4,
                                    batch_norm=False)

        # TODO: Define the generator network######

        #############################################

    def forward(self, x):
        # TODO: Define the forward pass#####

        out = F.relu(self.linear_bn(x))
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = torch.tanh(self.deconv3(out))
        return out


class CycleGenerator(nn.Module):
    def __init__(self, opts):
        super(CycleGenerator, self).__init__()
        self.opts = opts

        # TODO: Define the cyclegan generator network
        # self.generator_channels = [32, 64]
        self.conv1 = conv_layer(in_channels=3, out_channels=self.opts.generator_channels[0], kernel_size=4)
        self.conv2 = conv_layer(in_channels=self.opts.generator_channels[0],
                                out_channels=self.opts.generator_channels[1], kernel_size=4)
        self.resnet_block = ResNetBlock(self.opts.generator_channels[1])
        self.deconv1 = deconv_layer(in_channels=self.opts.generator_channels[1],
                                    out_channels=self.opts.generator_channels[0], kernel_size=4)
        self.deconv2 = deconv_layer(in_channels=self.opts.generator_channels[0], out_channels=3, kernel_size=4,
                                    batch_norm=False)

    def forward(self, x):
        # TODO: Define the forward pass
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block(out))

        out = F.relu(self.deconv1(out))
        out = torch.tanh(self.deconv2(out))
        return out
