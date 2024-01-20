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


# generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    # input:
    # layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    # large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    # small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].

    # output:
    # boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    # TODO:
    # create an numpy array "boxes" to store default bounding boxes
    # you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    boxes = np.zeros([10 * 10 + 5 * 5 + 3 * 3 + 1 * 1, 4, 8])
    # the first dimension means number of cells, 10*10+5*5+3*3+1*1
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
                    x_min, y_min = x_center - box_width / 2, y_center - box_height / 2
                    x_max, y_max = x_center + box_width / 2, y_center + box_height / 2
                    boxes[i * out_layer_size * out_layer_size + j * out_layer_size + k] = [x_center, y_center,
                                                                                           box_width, box_height, x_min,
                                                                                           y_min, x_max, y_max]

                    boxes = boxes.reshape([-1, 8])
                    # print("boxes:{}".format(boxes))
                    # print("boxes.shape:{}".format(boxes.shape))

    # the second dimension 4 means each cell has 4 default bounding boxes. their sizes are [ssize,ssize], [lsize,
    # lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)], where ssize is the corresponding size in
    # "small_scale" and lsize is the corresponding size in "large_scale". for a cell in layer[i], you should use
    # ssize=small_scale[i] and lsize=large_scale[i]. the last dimension 8 means each default bounding box has 8
    # attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]

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
    # input:
    # ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    # ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    # boxs_default            -- [num_of_boxes,8], default bounding boxes
    # threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    # cat_id                  -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box

    # compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min, y_min, x_max, y_max)

    ious_true = ious > threshold
    # TODO:
    # update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    # if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    # this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    gx = (x_max + x_min) / 2
    gy = (y_max + y_min) / 2
    gw = x_max - x_min
    gh = y_max - y_min

    for idx in range(len(ious_true)):
        if ious_true[idx]:
            tx = (gx - boxs_default[idx, 0]) / boxs_default[idx, 2]
            ty = (gy - boxs_default[idx, 1]) / boxs_default[idx, 3]
            tw = np.log(gw / boxs_default[idx, 2])
            th = np.log(gh / boxs_default[idx, 3])
            ann_box[idx] = [tx, ty, tw, th]
            ann_confidence[idx, cat_id] = 1
            ann_confidence[idx, -1] = 0

    ious_true = np.argmax(ious)
    # TODO:
    # make sure at least one default bounding box is used
    # update ann_box and ann_confidence (do the same thing as above)
    tx = (gx - boxs_default[ious_true, 0]) / boxs_default[ious_true, 2]
    ty = (gy - boxs_default[ious_true, 1]) / boxs_default[ious_true, 3]
    tw = np.log(gw / boxs_default[ious_true, 2])
    th = np.log(gh / boxs_default[ious_true, 3])
    ann_box[ious_true] = [tx, ty, tw, th]
    ann_confidence[ious_true, cat_id] = 1
    ann_confidence[ious_true, 3] = 0


def get_random_crop_coordinates(ann_data, width, height, margin=10):
    # 验证 ann_list 是否为空
    if len(ann_data) == 0:
        return 0, 0, width, height

    # 确保 margin 不会超过图像尺寸
    margin = min(margin, width // 2, height // 2)

    # 获取边界框的最小和最大坐标，并考虑边界
    min_x = max(np.floor(np.min(ann_data[:, 1])), margin)
    min_y = max(np.floor(np.min(ann_data[:, 2])), margin)
    max_x = min(np.ceil(np.max(ann_data[:, 3])), width - margin)
    max_y = min(np.ceil(np.max(ann_data[:, 4])), height - margin)

    # 计算裁剪坐标
    crop_x_min = random.randint(margin, min_x)
    crop_y_min = random.randint(margin, min_y)
    crop_x_max = random.randint(max_x, width - margin)
    crop_y_max = random.randint(max_y, height - margin)

    return crop_x_min, crop_y_min, crop_x_max, crop_y_max


class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train=True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num

        # overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)

        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        # notice:
        # you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

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

        # TODO:
        # 1. prepare the image [3,320,320], by reading image "img_name" first.
        img = cv2.imread(img_name)

        # (375, 500, 3)
        print(img.shape)
        # 2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        # 2
        # 317.29
        # 149.09
        # 126.63
        # 139.41
        ann_data = []
        with open(ann_name) as f:
            for l in f.readlines():
                print(l.split())
                class_id, x_min, y_min, width, height = l.split()
                x_max = float(x_min) + float(width)
                y_max = float(y_min) + float(height)
                ann_data.append([int(class_id), float(x_min), float(y_min), x_max, y_max])
        ann_data = np.array(ann_data)
        # 3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        # 4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        if self.train:

            width, height = img.shape[:2]
            crop_x_min, crop_y_min, crop_x_max, crop_y_max = get_random_crop_coordinates(ann_data, width, height)
            img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            width_new, height_new = img.shape[:2]
            image = cv2.resize(img, (self.image_size, self.image_size))

            ann_data[:, 1] = (ann_data[:, 1] - crop_x_min) / width_new
            ann_data[:, 3] = (ann_data[:, 3] - crop_x_min) / width_new
            ann_data[:, 2] = (ann_data[:, 2] - crop_y_min) / height_new
            ann_data[:, 4] = (ann_data[:, 4] - crop_y_min) / height_new
        else:
            image = cv2.resize(img, (self.image_size, self.image_size))
            ann_data[:, 1] = ann_data[:, 1] / width
            ann_data[:, 3] = ann_data[:, 3] / width
            ann_data[:, 2] = ann_data[:, 2] / height
            ann_data[:, 4] = ann_data[:, 4] / height
        for ann in ann_data:
            class_id, x_min, y_min, width, height = ann
            # print(class_id, x_min, y_min, width, height)
            match(ann_box, ann_confidence, self.boxs_default, self.threshold, int(class_id), x_min, y_min, width, height)
        # to use function "match":
        # match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        # where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.

        # note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        # For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        image = np.transpose(image, (2, 0, 1))
        return image, ann_box, ann_confidence
