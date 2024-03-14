# coding=utf-8

import gc
import math
import os
import pickle
import time
import collections
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision.ops import nms as NMS

def asMinutes(s):
    """
    Converts seconds to minutes and seconds format.

    Args:
        s: Number of seconds.

    Returns:
        Time in minutes and seconds format (string).
    """
    # Convert seconds to minutes
    m = math.floor(s / 60)
    # Calculate remaining seconds
    s -= m * 60
    # Return time in 'Xm Ys' format
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    """
    Calculates elapsed time and remaining time since a given starting time.

    Args:
        since: Starting time in seconds (e.g., time.time()).
        percent: Percentage of completion (0 to 1).

    Returns:
        Elapsed time and remaining time as a formatted string.
    """
    # Get current time
    now = time.time()
    # Calculate elapsed time in seconds
    s = now - since
    # Estimated total time
    es = s / (percent)
    # Remaining time
    rs = es - s
    # Format elapsed time and remaining time using asMinutes function
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def plot_concept_detector_results(
        image_to_save,
        image_orig,
        image_name,
        loc="upper right",
        bb_truth=None,
        bb_orig=None,
        bb_preds=None,
        dpi=500
):
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the transformed image in the first subplot
    ax[0].imshow(image_to_save, cmap=plt.cm.bone)

    # Add ground truth bounding box to the transformed image plot
    bounding_box = [bb_truth[0][0], bb_truth[0][1], bb_truth[0][2], bb_truth[0][3]]
    width = bounding_box[2] - bounding_box[0]
    height = bounding_box[3] - bounding_box[1]
    rect = patches.Rectangle(
        (bounding_box[0], bounding_box[1]),
        width,
        height,
        linewidth=3,
        edgecolor='r',
        facecolor='none',
        label='Ground Truth'
    )
    ax[0].add_patch(rect)
    ax[0].set_title('Transformed Image')
    ax[0].axis('off')

    # Add predicted bounding box to the transformed image plot if available
    if bb_preds is not None:
        bounding_box_pred = [bb_preds[0], bb_preds[1], bb_preds[2], bb_preds[3]]
        width_pred = bounding_box_pred[2] - bounding_box_pred[0]
        height_pred = bounding_box_pred[3] - bounding_box_pred[1]
        rect_pred = patches.Rectangle(
            (bounding_box_pred[0], bounding_box_pred[1]),
            width_pred,
            height_pred,
            linewidth=3,
            edgecolor='b',
            facecolor='none',
            label='Predicted'
        )
        ax[0].add_patch(rect_pred)

    # Add legend to the transformed image plot
    handles, labels = ax[0].get_legend_handles_labels()
    if handles:
        ax[0].legend(loc=loc, borderaxespad=0., framealpha=0.5)

    # Plot the original image in the second subplot
    ax[1].imshow(image_orig, cmap=plt.cm.bone)
    # Add ground truth bounding box to the original image plot
    bounding_box = [bb_orig[0], bb_orig[1], bb_orig[2], bb_orig[3]]
    width = bounding_box[2] - bounding_box[0]
    height = bounding_box[3] - bounding_box[1]
    rect = patches.Rectangle(
        (bounding_box[0], bounding_box[1]),
        width,
        height,
        linewidth=3,
        edgecolor='r',
        facecolor='none',
        label='Ground Truth'
    )
    ax[1].add_patch(rect)
    ax[1].set_title('Original Image')
    ax[1].axis('off')

    # Save the figure as an image file
    plt.savefig(image_name, bbox_inches='tight', dpi=dpi)
    
def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float - Array representing N bounding boxes with 4 coordinates (x1, y1, x2, y2)
    where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    b: (K, 4) ndarray of float - Array representing K bounding boxes with 4 coordinates (x1, y1, x2, y2)
    where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes a and b - Array containing the overlap between each pair of bounding boxes from a and b.
    """
    # Compute the area of each bounding box in array b
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    # Compute the width of intersection (iw) and height of intersection (ih) between bounding boxes in arrays a and b
    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    # Set negative values to 0 to ensure non-negative intersection
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    # Compute the union area (ua) of bounding boxes in arrays a and b
    ua = (np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih)

    # Ensure ua is not zero to avoid division by zero, set minimum value to floating-point epsilon
    ua = np.maximum(ua, np.finfo(float).eps)

    # Compute the intersection area between bounding boxes in arrays a and b
    intersection = iw * ih

    # Compute the overlap between bounding boxes in arrays a and b and return it
    return intersection / ua


def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    This function calculates the average precision as computed in py-faster-rcnn.

    Parameters:
        recall:    The recall curve (list).
        precision: The precision curve (list).

    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))  # Add sentinel values 0.0 and 1.0 to recall curve
    mpre = np.concatenate(([0.0], precision, [0.0]))  # Add sentinel values 0.0 and 0.0 to precision curve

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  # Update precision values with maximum value towards the end

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]  # Find where recall changes

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # Compute area under the precision-recall curve

    return ap


def _get_detections(dataset, retinanet, num_classes=1, score_threshold=0.05, max_detections=100):
    """
    Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(num_classes)] for j in range(len(dataset))]

    retinanet.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for index in range(len(dataset)):
            data = dataset[index]

            # run network
            scores, labels, boxes = retinanet(data["image"].to(device).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [
                        image_boxes,
                        np.expand_dims(image_scores, axis=1),
                        np.expand_dims(image_labels, axis=1),
                    ],
                    axis=1,
                )

                # copy detections to all_detections
                for label in range(num_classes):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(num_classes):
                    all_detections[index][label] = np.zeros((0, 5))

            print("{}/{}".format(index + 1, len(dataset)), end="\r")

    return all_detections


def _get_annotations(val_dataset, num_classes=11):
    """
    Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(num_classes)] for j in range(len(val_dataset))]
    for i in range(len(val_dataset)):
        # load the annotations
        annotations = val_dataset[i]['target']['boxes']

        # copy detections to all_annotations
        for label in range(num_classes):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].cpu().numpy().copy()

        print("{}/{}".format(i + 1, len(val_dataset)), end="\r")

    return all_annotations


class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).to(device)
        else:
            self.mean = mean

        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).to(device)
        else:
            self.std = std

    def forward(self, boxes, deltas):
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    @staticmethod
    def forward(boxes, img):
        """[clips the box coordinates]

        Args:
            boxes ([tensor]): [bounding box coordinates]
            img ([tensor]): [image tensor]

        Returns:
            [type]: [description]
        """
        # batch_size, num_channels, height, width = img.shape
        _, _, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


def compute_AUC(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
            true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
            can either be probability estimates of the positive class,
            confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    # Convert PyTorch tensors to numpy arrays
    gt_np = gt.cpu().numpy()  # Convert ground truth tensor to numpy array
    pred_np = pred.cpu().numpy()  # Convert prediction tensor to numpy array

    try:
        # Calculate AUROCs (Area Under the Receiver Operating Characteristic Curve) using sklearn
        AUROCs = roc_auc_score(gt_np, pred_np)

        # Calculate AUPRCs (Area Under the Precision-Recall Curve) using sklearn
        AUPRCs = average_precision_score(gt_np, pred_np)

    except:
        # If an error occurs during computation (e.g., all true labels are the same), set AUROCs and AUPRCs to 0.5
        AUROCs = 0.5
        AUPRCs = 0.5

    # Return the calculated AUROCs and AUPRCs
    return AUROCs, AUPRCs


def compute_accuracy(gt, pred):
    """
    Computes the accuracy given ground truth and predictions.

    Args:
        gt: PyTorch tensor or numpy array, true labels.
        pred: PyTorch tensor or numpy array, predicted labels.

    Returns:
        accuracy: Accuracy as a percentage.
    """
    # Count the number of correct predictions and divide by the total number of samples to get accuracy
    accuracy = (((pred == gt).sum()) / gt.size(0)).item() * 100
    return accuracy


def compute_auprc(gt, pred):
    """
    Computes the Area Under the Precision-Recall Curve (AUPRC) from ground truth and prediction scores.

    Args:
        gt: PyTorch tensor or numpy array, true binary labels.
        pred: PyTorch tensor or numpy array, probability estimates of the positive class.

    Returns:
        auprc: AUPRC score.
    """
    # Calculate AUPRC using sklearn's average_precision_score function
    auprc = average_precision_score(gt, pred)
    return auprc


def compute_accuracy_np_array(gt, pred):
    """
    Computes the accuracy given ground truth and predictions (numpy arrays).

    Args:
        gt: Numpy array, true labels.
        pred: Numpy array, predicted labels.

    Returns:
        accuracy: Accuracy as a percentage.
    """
    # Compute accuracy by comparing each element of gt and pred and then taking the mean
    accuracy = np.mean(gt == pred)
    return accuracy


def pr_auc(gt, pred, get_all=False):
    """
    Computes the Area Under the Precision-Recall Curve (PR AUC) from ground truth and prediction scores.

    Args:
        gt: Numpy array, true binary labels.
        pred: Numpy array, probability estimates of the positive class.
        get_all: Boolean, whether to return all computed values (precision, recall, and PR AUC).

    Returns:
        score: PR AUC score.
        precision: Precision values.
        recall: Recall values.
    """
    precision, recall, _ = precision_recall_curve(gt, pred)
    score = auc(recall, precision)
    if get_all:
        return score, precision, recall
    else:
        return score


def pfbeta(gt, pred, beta):
    """
    Computes the F-beta score given ground truth and prediction scores.

    Args:
        gt: Numpy array, true binary labels.
        pred: Numpy array, probability estimates of the positive class.
        beta: Float, beta value for F-beta score computation.

    Returns:
        result: F-beta score.
    """
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(gt)):
        prediction = min(max(pred[idx], 0), 1)
        if (gt[idx]):
            y_true_count += 1
            ctp += prediction
            # cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def auroc(gt, pred):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (AUROC) from ground truth and prediction scores.

    Args:
        gt: Numpy array, true binary labels.
        pred: Numpy array, probability estimates of the positive class.

    Returns:
        score: AUROC score.
    """
    return roc_auc_score(gt, pred)


def pfbeta_binarized(gt, pred):
    """
    Computes the maximum F-beta score given ground truth and prediction scores.

    Args:
        gt: Numpy array, true binary labels.
        pred: Numpy array, probability estimates of the positive class.

    Returns:
        max_score: Maximum F-beta score.
    """
    positives = pred[gt == 1]
    scores = []
    for th in positives:
        binarized = (pred >= th).astype('int')
        score = pfbeta(gt, binarized, 1)
        scores.append(score)

    return np.max(scores)


def nms(dets, thresh):
    """Non-Maximum Suppression (NMS)

    Args:
        dets (tensor): Tensor containing bounding boxes and corresponding scores.
        thresh (float): Threshold for NMS.

    Returns:
        tensor: Remaining bounding boxes after NMS.
    """
    # Extract bounding boxes and scores from the input tensor
    boxes = dets[:, :4]  # Extract bounding boxes (coordinates)
    scores = dets[:, -1]  # Extract scores

    # Apply Non-Maximum Suppression (NMS) to filter out redundant bounding boxes
    return NMS(boxes, scores, thresh)  # NMS function is called with bounding boxes, scores, and threshold


def get_paths(args):
    chk_pt_path = Path(f"{args.checkpoints}/{args.root}")
    output_path = Path(f"{args.output_path}/{args.root}")
    tb_logs_path = Path(f"{args.root}")

    return chk_pt_path, output_path, tb_logs_path
