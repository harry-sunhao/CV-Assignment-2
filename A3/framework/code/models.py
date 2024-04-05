import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_layer(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def deconv_layer(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv = conv_layer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, padding = 1, batch_norm = True)
    def forward(self, x):
        return x + self.conv(x)



class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        self.opts = opts
        conv_dim = opts.discriminator_channels
        #####TODO: Define the discriminator network#####
        self.conv1 = conv_layer(3, conv_dim[0], 4)  # Input is (32x32x3)
        self.conv2 = conv_layer(conv_dim[0], conv_dim[1], 4)  # (16x16x32)
        self.conv3 = conv_layer(conv_dim[1], conv_dim[2], 4)  # (8x8x64)
        self.conv4 = conv_layer(conv_dim[2], conv_dim[3], 4, stride=1, padding=0,batch_norm=False)  # (4x4x128)
    ################################################


    def forward(self, x):
        #####TODO: Define the forward pass#####
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x).squeeze()
        return x
        #######################################

class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.opts = opts
        conv_dim = opts.generator_channels
        z_size = opts.noise_size
        #####TODO: Define the generator network######
        self.deconv1 = deconv_layer(z_size, conv_dim[0], 4,stride=2, padding=0)
        self.deconv2 = deconv_layer(conv_dim[0], conv_dim[1], 4)
        self.deconv3 = deconv_layer(conv_dim[1], conv_dim[2], 4)
        self.deconv4 = deconv_layer(conv_dim[2], conv_dim[3], 4, batch_norm=False)
        #############################################

    def forward(self, x):
        #####TODO: Define the forward pass#####
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.tanh(self.deconv4(x))
        return x
        #######################################

class CycleGenerator(nn.Module):
    def __init__(self, opts):
        super(CycleGenerator, self).__init__()
        self.opts = opts

        #####TODO: Define the cyclegan generator network######
        g_conv_dim = opts.generator_channels
        # Encoder part of generator
        self.conv1 = conv_layer(3, g_conv_dim[0], 4)
        self.conv2 = conv_layer(g_conv_dim[0], g_conv_dim[1], 4)

        # Residual blocks

        self.res_blocks = ResNetBlock(g_conv_dim[1])
        # Decoder part of generator
        self.deconv2 = deconv_layer(g_conv_dim[1], g_conv_dim[0], 4)
        self.deconv3 = deconv_layer(g_conv_dim[0], 3, 4, batch_norm=False)
        ######################################################

    def forward(self, x):
        #####TODO: Define the forward pass#####
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Residual blocks
        x = F.relu(self.res_blocks(x))

        # Decoder
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        #######################################
        return x

