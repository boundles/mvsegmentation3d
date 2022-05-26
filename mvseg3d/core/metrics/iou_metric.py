import numpy as np
import torch
import torch.distributed as dist


class IOUMetric(object):
    """IOU Metric.

    Evaluate the result of the Semantic Segmentation.

    Args:
        id2label (dict): Map from label id to category name.
        ignore_index (int): Index that will be ignored in evaluation.
    """

    def __init__(self, id2label, ignore_index=255):
        self.id2label = id2label
        self.ignore_index = ignore_index

        self.hist_list = []

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

    def add(self, pred_labels, gt_labels):
        preds = pred_labels.clone().numpy().astype(np.int)
        labels = gt_labels.clone().numpy().astype(np.int)

        # filter out ignored points
        preds[preds == self.ignore_index] = -1
        labels[labels == self.ignore_index] = -1

        # calculate one instance result'
        hist = self.fast_hist(preds, labels, len(self.id2label))
        self.hist_list.append(hist)

    @staticmethod
    def reduce_tensor(tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def get_metric(self):
        hist = sum(self.hist_list)
        try:
            dist.barrier()
            hist = torch.from_numpy(hist).cuda()
            hist = self.reduce_tensor(hist).cpu().numpy()
            iou = self.per_class_iou(hist)
        except:
            iou = self.per_class_iou(hist)

        # mean iou
        metric = dict()
        miou = np.nanmean(iou)
        metric['mIOU'] = miou

        # iou per class
        iou_dict = dict()
        for i in range(len(self.id2label)):
            iou_dict[self.id2label[i]] = float(iou[i])
        metric['IOU'] = iou_dict
        return metric


if __name__ == '__main__':
    id2label = {0: 'bg', 1: 'fg'}
    iou_metric = IOUMetric(id2label, 255)

    pred_labels = torch.ones(3)
    gt_labels = torch.ones(3)
    iou_metric.add(pred_labels, gt_labels)

    pred_labels = torch.zeros(2)
    gt_labels = torch.zeros(2)
    iou_metric.add(pred_labels, gt_labels)
    print(iou_metric.get_metric())