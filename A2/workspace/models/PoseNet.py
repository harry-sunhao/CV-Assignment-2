import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle


def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        inception_1x1 = 'inception_{}/1x1'.format(key)
        inception_3x3_reduce = 'inception_{}/3x3_reduce'.format(key)
        inception_3x3 = 'inception_{}/3x3'.format(key)
        inception_5x5_reduce = 'inception_{}/5x5_reduce'.format(key)
        inception_5x5 = 'inception_{}/5x5'.format(key)
        inception_pool = 'inception_{}/pool_proj'.format(key)

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init(inception_1x1, nn.Conv2d(in_channels, n1x1, kernel_size=1), weights),
            nn.ReLU(True)
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            init(inception_3x3_reduce, nn.Conv2d(in_channels, n3x3red, kernel_size=1), weights),
            nn.ReLU(True),
            init(inception_3x3, nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1), weights),
            nn.ReLU(True)
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            init(inception_5x5_reduce, nn.Conv2d(in_channels, n5x5red, kernel_size=1), weights),
            nn.ReLU(True),
            init(inception_5x5, nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2), weights),
            nn.ReLU(True)
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.ReLU(True),
            init(inception_pool, nn.Conv2d(in_channels, pool_planes, kernel_size=1), weights),
            nn.ReLU(True)
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        return torch.cat((b1, b2, b3, b4), 1)


class LossHeader(nn.Module):
    def __init__(self, key, weights=None):
        super(LossHeader, self).__init__()

        # TODO: Define loss headers
        if key == 'loss1':
            inchannel = 512
        else:  # loss2
            inchannel = 528
        self.loss_red = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            init(key + '/conv', nn.Conv2d(inchannel, 128, kernel_size=1, stride=1), weights),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.7)
        )

        self.fc1 = nn.Linear(1024, 3)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        # TODO: Feed data through loss headers
        x = self.loss_red(x)

        xyz = self.fc1(x)
        wpqr = self.fc2(x)
        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            print(weights.keys())
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(True),

            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(True),

            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1, stride=1), weights),
            nn.ReLU(True),

            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), weights),
            nn.ReLU(True),

            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(True)

        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3b", weights)
        self._4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights)
        self._4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64, "4b", weights)
        self._4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64, "4c", weights)
        self._4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64, "4d", weights)
        self._4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128, "4e", weights)
        self._5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128, "5a", weights)
        self._5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128, "5b", weights)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.loss1 = LossHeader('loss1', weights)
        self.loss2 = LossHeader('loss2', weights)

        self.loss3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.Dropout(0.4)
        )

        self.out1 = nn.Linear(2048, 3)
        self.out2 = nn.Linear(2048, 4)

        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        x = self.pre_layers(x)

        x = self._3a(x)
        x = self._3b(x)

        x = self.maxpool1(x)

        x = self._4a(x)


        loss1_xyz, loss1_wpqr = self.loss1(x)

        x = self._4b(x)
        x = self._4c(x)
        x = self._4d(x)


        loss2_xyz, loss2_wpqr = self.loss2(x)

        x = self._4e(x)
        x = self.maxpool2(x)

        x = self._5a(x)
        x = self._5b(x)

        x = self.loss3(x)
        loss3_xyz = self.out1(x)
        loss3_wpqr = self.out2(x)
        if self.training:
            return loss1_xyz, \
                loss1_wpqr, \
                loss2_xyz, \
                loss2_wpqr, \
                loss3_xyz, \
                loss3_wpqr
        else:
            return loss3_xyz, \
                loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr

        self.weights_xyz = [w1_xyz, w2_xyz, w3_xyz]
        self.weights_wpqr = [w1_wpqr, w2_wpqr, w3_wpqr]
        self.losses_xyz = [nn.MSELoss() for _ in range(3)]
        self.losses_wpqr = [nn.MSELoss() for _ in range(3)]

    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        gt_xyz = poseGT[:, :3]
        gt_wpqr = poseGT[:, 3:]
        gt_norm_wpqr = torch.norm(gt_wpqr, dim=1, keepdim=True).clamp(min=1e-8)
        preds_xyz = [p1_xyz, p2_xyz, p3_xyz]
        preds_wpqr = [p1_wpqr, p2_wpqr, p3_wpqr]

        loss = 0.0
        for i in range(3):
            p_xyz, p_wpqr = preds_xyz[i], preds_wpqr[i]
            loss_xyz = self.losses_xyz[i](p_xyz, gt_xyz)
            loss_wpqr = self.losses_wpqr[i](p_wpqr, gt_wpqr / gt_norm_wpqr)
            loss += self.weights_xyz[i] * (loss_xyz + self.weights_wpqr[i] * loss_wpqr)

        return loss
