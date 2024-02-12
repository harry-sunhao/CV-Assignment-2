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
import numpy as np
import os
import cv2

import random

from utils import *


# generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    # input:
    # layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    # large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    # small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].

    # output:
    # boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.

    # create an numpy array "boxes" to store default bounding boxes
    # you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    boxes = []
    # the first dimension means number of cells, 10*10+5*5+3*3+1*1
    # default bounding box has a center (x=0.3, y=0.4), width=0.1 and height=0.2, the attributes you need to store are:
    # [0.3, 0.4, 0.1, 0.2, 0.25, 0.3, 0.35, 0.5]
    for i in range(len(layers)):
        out_layer_size = layers[i]
        ssize = small_scale[i]
        lsize = large_scale[i]
        # print("ssize:{},lsize:{},out_layer_size:{}".format(ssize,lsize,out_layer_size))
        boxes_sizes = [[ssize, ssize], [lsize, lsize], [lsize * np.sqrt(2), lsize / np.sqrt(2)],
                       [lsize / np.sqrt(2), lsize * np.sqrt(2)]]
        for j in range(out_layer_size):
            for k in range(out_layer_size):
                # print("j:{},k:{}".format(j,k))
                for box_size in boxes_sizes:
                    # print("box_size:{}".format(box_size))
                    x_center, y_center = (j + 0.5) / out_layer_size, (k + 0.5) / out_layer_size
                    box_width = box_size[0]
                    box_height = box_size[1]
                    box = create_bounding_box(x_center, y_center, box_width, box_height)[0]
                    boxes.append(box)
    boxes = np.array(boxes)
    return boxes


# this is an example implementation of IOU.
# It is different from the one used in YOLO, please pay attention.
# you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min, y_min, x_max, y_max):
    # input:
    # boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    # x_min,y_min,x_max,y_max -- another box (box_r)

    # output:
    # ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]

    inter = np.maximum(np.minimum(boxs_default[:, 6], x_max) - np.maximum(boxs_default[:, 4], x_min), 0) * np.maximum(
        np.minimum(boxs_default[:, 7], y_max) - np.maximum(boxs_default[:, 5], y_min), 0)
    area_a = (boxs_default[:, 6] - boxs_default[:, 4]) * (boxs_default[:, 7] - boxs_default[:, 5])
    area_b = (x_max - x_min) * (y_max - y_min)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)


def match(ann_box, ann_confidence, boxs_default, threshold, cat_id, x_min, y_min, x_max, y_max):
    # input: ann_box -- [num_of_boxes,4], ground truth bounding boxes to be updated
    # ann_confidence -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    # boxs_default   -- [num_of_boxes,8], default bounding boxes
    # threshold      -- if a default bounding box and the ground truth bounding box have iou>threshold,
    # then this default bounding box will be used as an anchor
    # cat_id         -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box

    # compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)

    ious_true = ious > threshold

    gx = (x_max + x_min) / 2
    gy = (y_max + y_min) / 2
    gw = x_max - x_min
    gh = y_max - y_min

    px, py, pw, ph = boxs_default[:, 0], boxs_default[:, 1], boxs_default[:, 2], boxs_default[:, 3]
    tx, ty, tw, th = (gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)

    if np.any(ious_true):
        tx_selected = tx[ious_true]
        ty_selected = ty[ious_true]
        tw_selected = tw[ious_true]
        th_selected = th[ious_true]

        ann_box[ious_true] = np.stack([tx_selected, ty_selected, tw_selected, th_selected], axis=1)
        ann_confidence[ious_true, cat_id] = 1
        ann_confidence[ious_true, -1] = 0
    else:
        max_iou_idx = np.argmax(ious)
        ann_box[max_iou_idx] = [tx[max_iou_idx], ty[max_iou_idx], tw[max_iou_idx], th[max_iou_idx]]
        ann_confidence[max_iou_idx, cat_id] = 1
        ann_confidence[max_iou_idx, -1] = 0


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train=True, image_size=320, debug=False, model="train"):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num

        self.debug = debug
        self.model = model
        # overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        self.img_names = os.listdir(self.imgdir)
        data_size = round(len(self.img_names) * 0.9)

        if self.model == "train":
            self.img_names = self.img_names[:data_size]
        elif self.model == "test":
            self.img_names = self.img_names[:]
        elif self.model == "val":
            self.img_names = self.img_names[data_size:]
        if self.debug:
            print(f'{model} dataset size: {len(self.img_names)}')
        self.image_size = image_size

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num, 4], np.float32)  # bounding boxes
        ann_confidence = np.zeros([self.box_num, self.class_num], np.float32)  # one-hot vectors
        # one-hot vectors with four classes
        # [1,0,0,0] -> cat
        # [0,1,0,0] -> dog
        # [0,0,1,0] -> person
        # [0,0,0,1] -> background

        ann_confidence[:, -1] = 1  # the default class for all cells is set to "background"

        img_name = self.imgdir + self.img_names[index]
        ann_name = self.anndir + self.img_names[index][:-3] + "txt"

        img = cv2.imread(img_name).astype(np.float32)
        height, width, img_c = img.shape

        if self.model == "test":
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = np.transpose(img, (2, 0, 1))
            return img, ann_box, ann_confidence
        annotations_txt = open(ann_name)
        annotations = annotations_txt.readlines()
        annotations_txt.close()

        for line in annotations:
            line = line.split()
            cat_id, x_min, y_min, w, h = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
            x_max = x_min + w
            y_max = y_min + h
            if self.model == "train":
                # Data  augmentation
                min_iou, max_iou = (0.3, float('inf'))
                for _ in range(50):
                    current_image = img
                    # aspect ratio constraint b/t .5 & 2
                    w = random.uniform(0.1 * width, width)
                    h = random.uniform(0.1 * height, height)
                    if h / w < 0.5 or h / w > 2:
                        continue
                    left = random.uniform(0, b=width - w)
                    top = random.uniform(0, b=height - h)

                    # convert to integer rect x1,y1,x2,y2
                    rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                    boxes = np.array([int(x_min), int(y_min), int(x_min + w), int(y_min + h)])

                    # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                    overlap = jaccard_numpy(boxes, rect)
                    # is min and max overlap constraint satisfied? if not try again
                    if overlap.min() < min_iou and max_iou < overlap.max():
                        continue
                    current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                    # keep overlap with gt box IF center in sampled patch
                    centers = (boxes[:2] + boxes[2:]) / 2.0

                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[0]) * (rect[1] < centers[1])

                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[0]) * (rect[3] > centers[1])

                    # mask in that both m1 and m2 are true
                    mask = m1 * m2

                    # have any valid boxes? try again if not
                    if not mask.any():
                        continue

                    # take only matching gt boxes
                    current_boxes = boxes.copy()

                    # should we use the box left and top corner or the crop's
                    current_boxes[:2] = np.maximum(current_boxes[:2], rect[:2])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:2] -= rect[:2]

                    current_boxes[2:] = np.minimum(current_boxes[2:], rect[2:])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[2:] -= rect[:2]
                    width = current_image.shape[1]
                    height = current_image.shape[0]
                    x_min = current_boxes[0]
                    y_min = current_boxes[1]
                    x_max = current_boxes[2]
                    y_max = current_boxes[3]
                    img = current_image.copy()
                    # print(f'x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}')
                    break

                # RandomBrightness
                if random.randint(0, 1):
                    delta = random.uniform(-32, 32)
                    img += delta
                # RandomContrast
                if random.randint(0, 1):
                    alpha = random.uniform(0.5, 1.5)
                    img *= alpha

            x_min = x_min / width
            y_min = y_min / height
            x_max = x_max / width
            y_max = y_max / height

            match(ann_box, ann_confidence, self.boxs_default, self.threshold, cat_id, x_min, y_min, x_max, y_max)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = np.transpose(img, (2, 0, 1))
        return img, ann_box, ann_confidence
