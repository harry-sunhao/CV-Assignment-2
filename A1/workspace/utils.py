import numpy as np
import cv2
from dataset import iou

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


# use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    # input:
    # windowname      -- the name of the window to display the images
    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    # image_          -- the input image to the network
    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]

    _, class_num = pred_confidence.shape
    # class_num = 4
    class_num = class_num - 1
    # class_num = 3 now, because we do not need the last class (background)

    image = np.transpose(image_, (1, 2, 0)).astype(np.uint8)
    image1 = np.zeros(image.shape, np.uint8)
    image2 = np.zeros(image.shape, np.uint8)
    image3 = np.zeros(image.shape, np.uint8)
    image4 = np.zeros(image.shape, np.uint8)
    image1[:] = image[:]
    image2[:] = image[:]
    image3[:] = image[:]
    image4[:] = image[:]
    # image1: draw ground truth bounding boxes on image1
    # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    # image3: draw network-predicted bounding boxes on image3
    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    width, height, _ = image.shape
    # draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i, j] > 0.5:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                # TODO:
                # image1: draw ground truth bounding boxes on image1
                # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)

                # ground truth bounding box
                gx = boxs_default[i, 2] * ann_box[i, 0] + boxs_default[i, 0]
                gy = boxs_default[i, 3] * ann_box[i, 1] + boxs_default[i, 1]
                gw = boxs_default[i, 2] * np.exp(ann_box[i, 2])
                gh = boxs_default[i, 3] * np.exp(ann_box[i, 3])

                image1_x1 = int(max(round((gx - 0.5 * gw) * width), 0))
                image1_y1 = int(max(round((gy - 0.5 * gh) * height), 0))
                image1_x2 = int(min(round((gx + 0.5 * gw) * width), width))
                image1_y2 = int(min(round((gy + 0.5 * gh) * height), height))
                image1_start_point = (image1_x1, image1_y1)  # top left corner, x1<x2, y1<y2
                image1_end_point = (image1_x2, image1_y2)

                image2_x1 = int(max(round(boxs_default[i, 4] * width), 0))
                image2_y1 = int(max(round(boxs_default[i, 5] * height), 0))
                image2_x2 = int(min(round(boxs_default[i, 6] * width), width))
                image2_y2 = int(min(round(boxs_default[i, 7] * height), height))
                image2_start_point = (image2_x1, image2_y1)  # top left corner, x1<x2, y1<y2
                image2_end_point = (image2_x2, image2_y2)

                color = colors[j]
                thickness = 2
                cv2.rectangle(image1, image1_start_point, image1_end_point, color, thickness)
                cv2.rectangle(image2, image2_start_point, image2_end_point, color, thickness)
                # you can use cv2.rectangle as follows:
                # start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                # end_point = (x2, y2) #bottom right corner
                # color = colors[j] #use red green blue to represent different classes
                # thickness = 2
                # cv2.rectangle(image?, start_point, end_point, color, thickness)

    # pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i, j] > 0.5:
                pass
                # TODO:
                # image3: draw network-predicted bounding boxes on image3
                # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                pred_box_ = np.zeros((pred_box.shape[0], 8), dtype=float)
                for i in range(pred_box_.shape[0]):
                    gx = boxs_default[i, 2] * pred_box[i, 0] + boxs_default[i, 0]
                    gy = boxs_default[i, 3] * pred_box[i, 1] + boxs_default[i, 1]
                    gw = boxs_default[i, 2] * np.exp(pred_box[i, 2])
                    gh = boxs_default[i, 3] * np.exp(pred_box[i, 3])

                    pred_box_[i, 0] = round(gx, 2)
                    pred_box_[i, 1] = round(gy, 2)
                    pred_box_[i, 2] = round(gw, 2)
                    pred_box_[i, 3] = round(gh, 2)
                    pred_box_[i, 4] = round(gx - 0.5 * gw, 2)
                    pred_box_[i, 5] = round(gy - 0.5 * gh, 2)
                    pred_box_[i, 6] = round(gx + 0.5 * gw, 2)
                    pred_box_[i, 7] = round(gy + 0.5 * gh, 2)

                image3_x1 = int(max(round(pred_box_[i, 4] * width), 0))
                image3_y1 = int(max(round(pred_box_[i, 5] * height), 0))
                image3_x2 = int(min(round(pred_box_[i, 6] * width), width))
                image3_y2 = int(min(round(pred_box_[i, 7] * height), height))
                image3_start_point = (image3_x1, image3_y1)  # top left corner, x1<x2, y1<y2
                image3_end_point = (image3_x2, image3_y2)

                # default box used
                image4_x1 = int(max(round(boxs_default[i, 4] * width), 0))
                image4_y1 = int(max(round(boxs_default[i, 5] * height), 0))
                image4_x2 = int(min(round(boxs_default[i, 6] * width), width))
                image4_y2 = int(min(round(boxs_default[i, 7] * height), height))

                image4_start_point = (image4_x1, image4_y1)  # top left corner, x1<x2, y1<y2
                image4_end_point = (image4_x2, image4_y2)
                color = colors[j]
                thickness = 2
                cv2.rectangle(image3, image3_start_point, image3_end_point, color, thickness)
                cv2.rectangle(image4, image4_start_point, image4_end_point, color, thickness)

    # combine four images into one
    h, w, _ = image1.shape
    image = np.zeros([h * 2, w * 2, 3], np.uint8)
    image[:h, :w] = image1
    image[:h, w:] = image2
    image[h:, :w] = image3
    image[h:, w:] = image4
    # cv2.imshow(windowname + " [[gt_box,gt_dft],[pd_box,pd_dft]]", image)
    # cv2.waitKey(1)
    save_name = f'{windowname}.png'
    cv2.imwrite(filename=save_name, img=image)

    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization.


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    # TODO: non maximum suppression
    # input:
    # confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # boxs_default -- default bounding boxes, [num_of_boxes, 8]
    # overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    # threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.

    # output:
    # depends on your implementation.
    # if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    # you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    pass


def update_precision_recall(pred_confidence, pred_box, ann_confidence, ann_box, boxs_default, precision, recall,
                            threshold):
    # pred_confidence: [num_of_boxes, num_of_classes] 的预测置信度
    # pred_box: [num_of_boxes, 4] 的预测边界框
    # ann_confidence: [num_of_boxes, num_of_classes] 的真实类别标签
    # ann_box: [num_of_boxes, 4] 的真实边界框
    # boxs_default: [num_of_boxes, 8] 默认的边界框
    # precision: 精度的累积列表
    # recall: 召回率的累积列表
    # threshold: 置信度阈值

    # 通过阈值筛选出置信度较高的预测
    # 这里我们假设预测和真实数据都是单类别的，所以我们只取最大置信度的索引
    pred_labels = pred_confidence.max(1)
    pred_indices = pred_labels > threshold

    # 选出对应的预测框和标注框
    pred_box_filtered = pred_box[pred_indices]
    ann_box_filtered = ann_box[pred_indices]

    # 初始化TP, FP, FN计数器
    TP = 0
    FP = 0
    FN = 0

    # 对于每个预测框，判断是否为TP或FP
    for i in range(len(pred_box_filtered)):
        iou_score = iou(pred_box_filtered[i], ann_box_filtered[i])
        if iou_score > threshold:
            TP += 1
        else:
            FP += 1

    # 计算FN
    FN = len(ann_confidence) - TP

    # 计算精度和召回率
    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0

    # 更新精度和召回率列表
    precision.append(prec)
    recall.append(rec)

    # 可以返回更新后的precision和recall列表，或者返回当前的prec和rec
    # return precision, recall
    return prec, rec


def generate_mAP():
    # TODO: Generate mAP
    pass
