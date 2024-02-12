import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


# def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
#
#     pos_mask = ann_confidence[..., -1] == 0
#     neg_mask = ann_confidence[..., -1] == 1
#
#     pos_class_loss = F.cross_entropy(pred_confidence[pos_mask], ann_confidence[pos_mask].argmax(1), reduction='sum')
#
#     neg_class_loss = F.cross_entropy(pred_confidence[neg_mask], ann_confidence[neg_mask].argmax(1), reduction='sum')
#
#     bbox_loss = F.smooth_l1_loss(pred_box[pos_mask], ann_box[pos_mask], reduction='sum')
#
#     num_pos = pos_mask.sum().float()
#     num_neg = neg_mask.sum().float()
#     total_loss = pos_class_loss / num_pos + 3 * neg_class_loss / num_neg + bbox_loss / num_pos
#
#     return total_loss

def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    pass
    # input:
    # pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    # output:
    # loss -- a single number for the value of the loss function, [1]

    # TODO: write a loss function for SSD
    #
    # For confidence (class labels), use cross entropy (F.cross_entropy)
    # You can try F.binary_cross_entropy and see which loss is better
    # For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    # Note that you need to consider cells carrying objects and empty cells separately.
    # I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    # and reshape box to [batch_size*num_of_boxes, 4].

    batch_size, num_of_boxes, num_of_classes = pred_confidence.shape

    pred_confidence = torch.reshape(pred_confidence, (batch_size * num_of_boxes, num_of_classes))
    ann_confidence = torch.reshape(ann_confidence, (batch_size * num_of_boxes, num_of_classes))
    pred_box = torch.reshape(pred_box, (batch_size * num_of_boxes, 4))
    ann_box = torch.reshape(ann_box, (batch_size * num_of_boxes, 4))
    # Then you need to figure out how you can get the indices of all cells carrying objects,
    # and use confidence[indices], box[indices] to select those cells.

    non_empty_ann_confidence = ann_confidence[ann_confidence[:, 3] != 1]
    non_empty_pred_confidence = pred_confidence[ann_confidence[:, 3] != 1]
    non_empty_bbox = ann_box[ann_confidence[:, 3] != 1]
    non_empty_pred_bbox = pred_box[ann_confidence[:, 3] != 1]
    empty_ann_confidence = ann_confidence[ann_confidence[:, 3] == 1]
    empty_pred_confidence = pred_confidence[ann_confidence[:, 3] == 1]

    # Compute Loss
    class_loss = F.cross_entropy(non_empty_pred_confidence, non_empty_ann_confidence) + 3 * F.cross_entropy(
        empty_ann_confidence, empty_pred_confidence)
    bbox_loss = F.smooth_l1_loss(non_empty_pred_bbox, non_empty_bbox)
    loss = class_loss + bbox_loss
    return loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()

        self.class_num = class_num  # num_of_classes, in this assignment, 4: cat, dog, person, background

        # TODO: define layers
        self.BaseBlock = nn.Sequential(
            ConvBlock(3, 64, 3, 2, 1),  # 320 - 160
            ConvBlock(64, 64, 3, 1, 1),  # 160 -160
            ConvBlock(64, 64, 3, 1, 1),  # 160 - 160
            ConvBlock(64, 128, 3, 2, 1),  # 160 -80
            ConvBlock(128, 128, 3, 1, 1),  # 80 -80
            ConvBlock(128, 128, 3, 1, 1),  # 80 -80
            ConvBlock(128, 256, 3, 2, 1),  # 80 -40
            ConvBlock(256, 256, 3, 1, 1),  # 40 -40
            ConvBlock(256, 256, 3, 1, 1),  # 40 -40
            ConvBlock(256, 512, 3, 2, 1),  # 40 -20
            ConvBlock(512, 512, 3, 1, 1),  # 20 -20
            ConvBlock(512, 512, 3, 1, 1),  # 20 -20
            ConvBlock(512, 256, 3, 2, 1)  # 20 -10
        )
        self.batch1 = nn.Sequential(
            ConvBlock(256, 256, 1, 1, 0),
            ConvBlock(256, 256, 3, 2, 1)  # 10 - 5
        )
        self.batch2 = nn.Sequential(
            ConvBlock(256, 256, 1, 1, 0),
            ConvBlock(256, 256, 3, 1, 0)  # 5 - 3
        )
        self.batch3 = nn.Sequential(
            ConvBlock(256, 256, 1, 1, 0),
            ConvBlock(256, 256, 3, 1, 0)  # 3 - 1
        )
        self.conv3x3 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        # input:
        # x -- images, [batch_size, 3, 320, 320]

        x = x / 255.0  # normalize image. If you already normalized your input image in the dataloader, remove this line.

        # TODO: define forward

        x = self.BaseBlock(x)
        box1 = self.conv3x3(x)
        box1 = torch.reshape(box1, (-1, 16, 100))
        confidence1 = self.conv3x3(x)
        confidence1 = torch.reshape(confidence1, (-1, self.class_num * 4, 100))

        x = self.batch1(x)
        box2 = self.conv3x3(x)
        box2 = torch.reshape(box2, (-1, 16, 25))
        confidence2 = self.conv3x3(x)
        confidence2 = torch.reshape(confidence2, (-1, self.class_num * 4, 25))

        x = self.batch2(x)
        box3 = self.conv3x3(x)
        box3 = torch.reshape(box3, (-1, 16, 9))
        confidence3 = self.conv3x3(x)
        confidence3 = torch.reshape(confidence3, (-1, self.class_num * 4, 9))

        x = self.batch3(x)
        box4 = self.conv3x3(x)
        box4 = torch.reshape(box4, (-1, 16, 1))
        confidence4 = self.conv3x3(x)
        confidence4 = torch.reshape(confidence4, (-1, self.class_num * 4, 1))

        bboxes = torch.cat((box1, box2, box3, box4), 2)
        confidence = torch.cat((confidence1, confidence2, confidence3, confidence4), 2)

        bboxes = bboxes.permute(0, 2, 1)
        confidence = confidence.permute(0, 2, 1)

        bboxes = torch.reshape(bboxes, (-1, 540, 4))
        confidence = torch.reshape(confidence, (-1, 540, self.class_num))

        # print(bboxes.shape)
        # print(confidence.shape)
        confidence = F.softmax(confidence, dim=2)

        # should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?

        # sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        # confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        # bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]

        return confidence, bboxes
