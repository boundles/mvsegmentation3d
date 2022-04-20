import numpy as np
import torch


class IOUMetric(object):
    """IOU Metric.

    Evaluate the result of the Semantic Segmentation.

    Args:
        label2cat (dict): Map from label to category name.
        ignore_index (int): Index that will be ignored in evaluation.
    """

    def __init__(self, label2cat, ignore_index=255):
        self.label2cat = label2cat
        self.ignore_index = ignore_index

        self.pred_labels = []
        self.gt_labels = []

    @staticmethod
    def fast_hist(preds, labels, num_classes):
        """Compute the confusion matrix for every batch.
        Args:
            preds (np.ndarray):  Prediction labels of points with shape of
            (num_points, ).
            labels (np.ndarray): Ground truth labels of points with shape of
            (num_points, ).
            num_classes (int): number of classes
        Returns:
            np.ndarray: Calculated confusion matrix.
        """

        k = (labels >= 0) & (labels < num_classes)
        bin_count = np.bincount(
            num_classes * labels[k].astype(int) + preds[k],
            minlength=num_classes ** 2)
        return bin_count[:num_classes ** 2].reshape(num_classes, num_classes)

    @staticmethod
    def per_class_iou(hist):
        """Compute the per class iou.
        Args:
            hist(np.ndarray):  Overall confusion martix
            (num_classes, num_classes ).
        Returns:
            np.ndarray: Calculated per class iou
        """

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def add(self, pred_label, gt_label):
        self.pred_labels.append(pred_label)
        self.gt_labels.append(gt_label)

    def get_metric(self):
        assert len(self.pred_labels) == len(self.gt_labels)
        num_classes = len(self.label2cat)

        hist_list = []
        for i in range(len(self.gt_labels)):
            pred_label = self.pred_labels[i].clone().numpy().astype(np.int)
            gt_label = self.gt_labels[i].clone().numpy().astype(np.int)

            # filter out ignored points
            pred_label[pred_label == self.ignore_index] = -1
            gt_label[gt_label == self.ignore_index] = -1

            # calculate one instance result
            hist_list.append(self.fast_hist(pred_label, gt_label, num_classes))

        iou = self.per_class_iou(sum(hist_list))
        miou = np.nanmean(iou)

        # mean iou
        metric = dict()
        metric['mIOU'] = miou

        # iou per class
        iou_dict = dict()
        for i in range(len(self.label2cat)):
            iou_dict[self.label2cat[i]] = float(iou[i])
        metric['IOU'] = iou_dict
        return metric


if __name__ == '__main__':
    label2cat = {}
    label2cat[0] = 'bg'
    label2cat[1] = 'fg'
    iou_metric = IOUMetric(label2cat, 255)

    pred_label = torch.ones(3)
    gt_label = torch.ones(3)
    iou_metric.add(pred_label, gt_label)

    pred_label = torch.zeros(2)
    gt_label = torch.zeros(2)
    iou_metric.add(pred_label, gt_label)
    print(iou_metric.get_metric())