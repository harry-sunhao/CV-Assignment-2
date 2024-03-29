import numpy as np
import cv2
from matplotlib import pyplot as plt

import dataset

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


# use [blue green red] to represent different classes
def denormalization(boxes, width, height):
    boxes_np = np.array(boxes)
    if boxes_np.ndim == 1:
        if len(boxes) == 4:
            scale = np.tile([width, height], 2)
        elif len(boxes) == 8:
            scale = np.tile([width, height], 2)
        else:
            scale = np.tile([])
    else:
        if boxes_np.shape[1] == 4:
            scale = np.tile([width, height], 2)
        elif boxes_np.shape[1] == 8:
            scale = np.tile([width, height], 4)
        else:
            scale = np.tile([])
    denormalized_boxes = boxes * scale
    return denormalized_boxes.astype(int)


def get_coords(center_x, center_y, width, height):
    half_width, half_height = width / 2.0, height / 2.0
    top_left_x = np.clip(center_x - half_width, 0, 1)
    top_left_y = np.clip(center_y - half_height, 0, 1)
    bottom_right_x = np.clip(center_x + half_width, 0, 1)
    bottom_right_y = np.clip(center_y + half_height, 0, 1)
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def create_bounding_box(center_x, center_y, width, height):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = get_coords(center_x, center_y, width, height)
    # Concatenate all values into a single array
    box = np.column_stack((center_x, center_y, width, height,
                           top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    return np.around(box, 2)


def recover_single_gt_box(single_box, box_default):
    gx = box_default[2] * single_box[0] + box_default[0]
    gy = box_default[3] * single_box[1] + box_default[1]
    gw = box_default[2] * np.exp(single_box[2])
    gh = box_default[3] * np.exp(single_box[3])

    return gx, gy, gw, gh


def recover_original_boxes(boxes, boxs_default):
    actual_boxes = np.zeros_like(boxes)
    for i in range(len(boxes)):
        actual_boxes[i, 0], actual_boxes[i, 1], actual_boxes[i, 2], actual_boxes[i, 3] = recover_single_gt_box(boxes[i],
                                                                                                               boxs_default[
                                                                                                                   i])
    return create_bounding_box(actual_boxes[:, 0], actual_boxes[:, 1], actual_boxes[:, 2], actual_boxes[:, 3])


def draw_bounding_box(class_num, width, height, confidences, box, boxs_default, image1, image2, thickness):
    for i in range(len(confidences)):
        for j in range(class_num):
            if confidences[i, j] > 0.5:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                # [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
                # [0,           1       ,2          ,3,         4,      5,      6,     7]
                color = colors[j]
                # ground truth bounding box
                gx, gy, gw, gh = recover_single_gt_box(box[i], boxs_default[i])

                img1_coordi = denormalization(get_coords(gx, gy, gw, gh), width, height)
                img2_coordi = denormalization(boxs_default[i, 4:8], width, height)
                # [x1 y1 x2 y2]

                cv2.rectangle(image1, (img1_coordi[0], img1_coordi[1]), (img1_coordi[2], img1_coordi[3]), color,
                              thickness)
                cv2.rectangle(image2, (img2_coordi[0], img2_coordi[1]), (img2_coordi[2], img2_coordi[3]), color,
                              thickness)
    return image1, image2


def visualize_pred_train(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default,
                         pred_boxes_default=None):
    # input:
    # windowname      -- the name of the window to display the images
    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    # image_          -- the input image to the network
    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]

    # _, class_num = pred_confidence.shape
    class_num = 4
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
    thickness = 2
    # draw ground truth
    image1, image2 = draw_bounding_box(class_num, width, height, ann_confidence, ann_box, boxs_default, image1, image2,
                                       thickness)
    image3, image4 = draw_bounding_box(class_num, width, height, pred_confidence, pred_box, boxs_default, image3,
                                       image4, thickness)

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


def visualize_pred_custom(windowname, pred_boxes, pred_confidence, default_boxes, image_):
    # input:
    # windowname      -- the name of the window to display the images
    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    # image_          -- the input image to the network
    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]

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

    height, width, _ = image.shape

    pred_boxes = denormalization(pred_boxes, width, height)
    default_boxes = denormalization(default_boxes, width, height)
    class_ids = np.argmax(pred_confidence, axis=1)
    thickness = 2

    # pred
    for i in range(len(pred_boxes)):
        color = colors[class_ids[i]]
        image3 = cv2.rectangle(image3, (pred_boxes[i, 4], pred_boxes[i, 5]), (pred_boxes[i, 6], pred_boxes[i, 7]),
                               color, thickness)
        image4 = cv2.rectangle(image4, (default_boxes[i, 4], default_boxes[i, 5]),
                               (default_boxes[i, 6], default_boxes[i, 7]), color, thickness)

    # combine four images into one
    height, width, _ = image1.shape
    image = np.zeros([height * 2, width * 2, 3], np.uint8)
    image[:height, :width] = image1
    image[:height, width:] = image2
    image[height:, :width] = image3
    image[height:, width:] = image4

    cv2.imwrite(f'{windowname}.png', image)


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    # TODO: non maximum suppression input: confidence_  -- the predicted class labels from SSD, [num_of_boxes,
    #  num_of_classes] box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4] boxs_default --
    #  default bounding boxes, [num_of_boxes, 8] overlap      -- if two bounding boxes in the same class have iou >
    #  overlap, then one of the boxes must be suppressed threshold    -- if one class in one cell has confidence >
    #  threshold, then consider this cell carrying a bounding box with this class.

    # output: depends on your implementation. if you wish to reuse the visualize_pred_train function above,
    # you need to return a "suppressed" version of confidence [5,5, num_of_classes]. you can also directly return the
    # final bounding boxes and classes, and write a new visualization function for that.

    # Recover the original bounding boxes from predictions
    boxes = recover_original_boxes(box_, boxs_default)

    # Make copies of the confidence scores and default boxes
    confidences = confidence_.copy()
    default_boxes = boxs_default.copy()

    # Lists to store the selected bounding boxes, their confidence scores, and default boxes
    selected_boxes = []
    selected_confidences = []
    selected_boxs_default = []

    # NMS Loop: Iterate as long as there are bounding boxes in confidences
    while confidences.shape[0] > 0:
        # Find the index of the box with the highest confidence,
        max_confidence_index = np.argmax(confidences[:, :-1], axis=None)
        max_box_index = max_confidence_index // (confidences.shape[1] - 1)

        # Break the loop if the highest confidence is below the threshold
        if confidences[max_box_index, :-1].max() < threshold:
            break

        # Save the selected bounding box's information before removal
        selected_boxes.append(boxes[max_box_index])
        selected_confidences.append(confidences[max_box_index, :-1])  # Exclude the last class
        selected_boxs_default.append(default_boxes[max_box_index])

        # Retrieve coordinates of the selected bounding box
        selected_box = default_boxes[max_box_index, 4:8]

        # Compute IOU with the remaining boxes
        remaining_iou = dataset.iou(default_boxes, *selected_box)
        mask = remaining_iou <= overlap  # Boolean mask to filter out boxes with high IOU

        # Update the lists by filtering out the selected box and boxes with high overlap
        confidences = confidences[mask]
        boxes = boxes[mask]
        default_boxes = default_boxes[mask]

    # Convert the lists of selected boxes, confidences, and default boxes to arrays
    final_boxes = np.array(selected_boxes)
    final_confidence = np.array(selected_confidences)
    final_boxs_default = np.array(selected_boxs_default)

    # Return the final selected bounding boxes, their confidences, and default boxes
    return final_boxes, final_confidence, final_boxs_default


def bulid_class_boxes(confidences, boxes, num_classes, pred=False):
    class_boxes = [[] for _ in range(num_classes)]
    class_scores = [[] for _ in range(num_classes)]
    for confidence, box in zip(confidences, boxes):
        class_id = np.argmax(confidence)
        if class_id >= num_classes:
            continue
        if not pred:
            pred_score = 0
        else:
            pred_score = np.max(confidence)
        class_boxes[class_id].append(box)
        class_scores[class_id].append(pred_score)
    return class_boxes, class_scores


def update_precison_recall(pred_box, pred_confidence, ann_box, ann_confidence,
                           boxes_default,
                           num_classes, iou_threshold=0.5):
    # num_classes is true number of class is 3

    gt_boxes = recover_original_boxes(ann_box, boxes_default)

    class_type = [[] for _ in range(num_classes)]

    class_true_boxes, _ = bulid_class_boxes(confidences=ann_confidence, boxes=gt_boxes,
                                            num_classes=num_classes, pred=False)
    class_predicted_boxes, class_predicted_scores = bulid_class_boxes(confidences=pred_confidence, boxes=pred_box,
                                                                      num_classes=num_classes, pred=True)
    class_counts = [len(class_true_boxes[i]) for i in range(num_classes)]

    for i in range(num_classes):
        if class_true_boxes[i]:
            length = len(class_predicted_boxes[i])
            # sort by scores
            sorted_indices = np.argsort(class_predicted_scores[i])
            class_predicted_boxes[i] = np.array(class_predicted_boxes[i])[sorted_indices].tolist()
            class_predicted_scores[i] = np.array(class_predicted_scores[i])[sorted_indices].tolist()

            gt_matched = np.zeros(len(class_true_boxes[i]), dtype=bool)
            true_boxes_np = np.array(class_true_boxes[i])
            for idx, pred_box_idx in enumerate(class_predicted_boxes[i]):
                ious = dataset.iou(true_boxes_np, *pred_box_idx[4:8])
                best_iou_index = np.argmax(ious)
                best_iou = ious[best_iou_index]
                if best_iou > iou_threshold and not gt_matched[best_iou_index]:
                    gt_matched[best_iou_index] = True
                    class_type[i].append({"score": class_predicted_scores[i][idx], "type": 'TP'})
                else:
                    class_type[i].append({"score": class_predicted_scores[i][idx], "type": 'FP'})
    return class_counts, class_type


def generate_mAP(class_counts, cumulative_TPs, cumulative_FPs):
    list_AP = []
    pr_curves = []
    arr0 = np.array([0])
    arr1 = np.array([1])
    num_classes = len(class_counts)

    for i in range(num_classes):
        cumulative_TPs[i].sort(key=lambda pred: pred["score"], reverse=True)
        cumulative_TP = np.zeros(len(cumulative_TPs[i]))
        cumulative_FP = np.zeros(len(cumulative_TPs[i]))
        for idx, P_type in enumerate(cumulative_TPs[i]):
            if P_type["type"] == "TP":
                cumulative_TP[idx] = 1
            elif P_type["type"] == "FP":
                cumulative_FP[idx] = 1
        # tp_cumsum = np.cumsum(cumulative_TPs[i])
        # fp_cumsum = np.cumsum(cumulative_FPs[i])
        tp_cumsum = np.cumsum(cumulative_TP)
        fp_cumsum = np.cumsum(cumulative_FP)
        total_gt = class_counts[i]
        recalls = tp_cumsum / np.maximum(total_gt, 1e-8)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)
        # By adding these two points, the PR curve starts at (0, 1), which is a recall of 0 and a precision of 1.
        # This helps in calculating the AP value correctly since AP is the area under the PR curve and this area
        # should start from (0, 1) and end at the actual data point.
        #
        # This practice ensures accuracy and completeness when calculating the area under the curve (such as using
        # the np.trapz function). Without these two points, the PR curve would start directly from the first point in
        # the data, potentially leading to an inaccurate estimate of model performance.

        precisions = np.concatenate((arr1, precisions))
        recalls = np.concatenate((arr0, recalls))

        list_AP.append(np.trapz(precisions, recalls))
        pr_curves.append((precisions, recalls))

    return np.mean(list_AP), pr_curves


def plot_precision_recall_curves(pr_curves, num_classes):
    plt.figure(figsize=(12, 8))
    for i in range(num_classes):
        precisions, recalls = pr_curves[i]
        plt.plot(recalls, precisions, label=f'Class {i + 1}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[2:], box_b[2:])
    min_xy = np.maximum(box_a[:2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[0] * inter[1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[2] - box_a[0]) *
              (box_a[3] - box_a[1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
