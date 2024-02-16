import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *


def main(args):
    # please google how to use argparse
    # a short intro:
    # to train: python main.py
    # to test:  python main.py --test

    class_num = 4  # cat dog person background

    num_epochs = 200
    batch_size = 64
    num_workers = 0

    boxs_default = default_box_generator([10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7])

    assert boxs_default.shape == (540, 8)

    assert np.array_equal(boxs_default[0], np.array([0.05, 0.05, 0.1, 0.1, 0, 0, 0.1, 0.1]))
    assert np.array_equal(boxs_default[1], np.array([0.05, 0.05, 0.2, 0.2, 0, 0, 0.15, 0.15]))
    assert np.array_equal(boxs_default[2], np.array([0.05, 0.05, 0.28, 0.14, 0, 0, 0.19, 0.12]))
    assert np.array_equal(boxs_default[3], np.array([0.05, 0.05, 0.14, 0.28, 0, 0, 0.12, 0.19]))

    # Create network
    network = SSD(class_num)
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)

    cudnn.benchmark = True
    cumulative_type = [[] for _ in range(class_num)]
    cumulative_FPs = [[] for _ in range(class_num)]
    class_count = [0 for _ in range(class_num - 1)]

    if not args.test:
        dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, model="train",
                       image_size=320, debug=args.debug,data_augmentation=args.data_augmentation)
        dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, model="val",
                            image_size=320, debug=args.debug)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True,
                                                      num_workers=num_workers)

        # optimizer = optim.Adam(network.parameters(), lr=1e-4)
        optimizer = optim.RMSprop(network.parameters(), lr=1e-4, alpha=0.9)
        # feel free to try other optimizers and parameters.

        start_time = time.time()

        for epoch in range(num_epochs):
            # TRAINING
            network.train()

            avg_loss = 0
            avg_count = 0
            for i, data in enumerate(dataloader, 0):
                images_, ann_box_, ann_confidence_ = data
                images = images_.cuda()
                ann_box = ann_box_.cuda()
                ann_confidence = ann_confidence_.cuda()

                optimizer.zero_grad()
                pred_confidence, pred_box = network(images)
                loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
                loss_net.backward()
                optimizer.step()

                avg_loss += loss_net.data
                avg_count += 1

            print('[%d] time: %f train loss: %f' % (
                epoch, time.time() - start_time, avg_loss / avg_count))

            # visualize
            if args.save:
                pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
                pred_box_ = pred_box[0].detach().cpu().numpy()
                visualize_pred_train(f'train_{epoch}', pred_confidence_, pred_box_, ann_confidence_[0].numpy(),
                                     ann_box_[0].numpy(),
                                     images_[0].numpy(), boxs_default)

            # VALIDATION
            network.eval()

            # TODO: split the dataset into 90% training and 10% validation
            # use the training set to train and the validation set to evaluate

            for i, data in enumerate(dataloader_test, 0):
                images_, ann_box_, ann_confidence_ = data
                images = images_.cuda()

                pred_confidence, pred_box = network(images)

                pred_confidence_ = pred_confidence.detach().cpu().numpy()
                pred_box_ = pred_box.detach().cpu().numpy()
                for j in range(pred_confidence_.shape[0]):
                    single_image_pred_confidence = pred_confidence_[j]
                    single_image_pred_box = pred_box_[j]
                    single_image_gt_boxes = ann_box_[j].cpu().numpy()
                    single_image_gt_confidence = ann_confidence_[j].cpu().numpy()

                    single_image_pred_box_nms, single_image_pred_confidence_nms, single_image_boxs_default_nms = non_maximum_suppression(
                        single_image_pred_confidence, single_image_pred_box, boxs_default)

                    gt_count_by_single, precision_by_class = update_precison_recall(
                        single_image_pred_box_nms, single_image_pred_confidence_nms,
                        single_image_pred_box,
                        single_image_gt_confidence, boxs_default,
                        class_num - 1,
                        iou_threshold=0.5)
                    for c in range(class_num - 1):
                        class_count[c] += gt_count_by_single[c]
                        cumulative_type[c].extend(precision_by_class[c])
            if args.save:
                visualize_pred_train(f'val_{epoch}', single_image_pred_confidence_nms, single_image_pred_box_nms,
                                     single_image_gt_confidence, single_image_gt_boxes, images_[j].numpy(),
                                     boxs_default, single_image_boxs_default_nms)

            # save weights
            if epoch % 10 == 9:
                # save last network
                print('saving net...')
                torch.save(network.state_dict(), 'network.pth')
        mAP, pr_curves = generate_mAP(class_count, cumulative_type, cumulative_FPs)
        plot_precision_recall_curves(pr_curves, class_num - 1)

        print(f'mAP:{mAP * 100}%')

    else:
        # TEST
        dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, model="test",
                            image_size=320, debug=args.debug)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
        network.load_state_dict(torch.load(args.test_path))
        network.eval()

        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()

            pred_confidence, pred_box = network(images)

            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()

            suppressed_boxes, suppressed_confidence, corresponding_default_boxes = non_maximum_suppression(
                pred_confidence_,
                pred_box_, boxs_default)
            # TODO: save predicted bounding boxes and classes to a txt file.
            # you will need to submit those files for grading this assignment
            if len(suppressed_boxes) > 0:
                if args.debug:
                    print(f'test_{i}: {suppressed_boxes}')
                visualize_pred_custom(f'test_{i}', suppressed_boxes, suppressed_confidence, corresponding_default_boxes,
                                      images_[0].numpy())
                print(f'write test_{i}...')

            cv2.waitKey(1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_path', type=str, default='network.pth')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--data_augmentation', action='store_true')
    args = parser.parse_args()
    # args.test = False
    # args.test_path = 'network_200.pth'
    main(args)
