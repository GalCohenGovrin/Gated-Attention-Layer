# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class evalMetrics(object):
    """Computes and stores the evaluation metrics"""

    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val_dice_liver = 0
        self.val_precision_liver = 0
        self.val_recall_liver = 0
        
        self.val_dice_lesion = 0
        self.val_precision_lesion = 0
        self.val_recall_lesion = 0
        
        self.avg_dice_liver = 0
        self.avg_precision_liver = 0
        self.avg_recall_liver = 0
        
        self.avg_dice_lesion = 0
        self.avg_precision_lesion = 0
        self.avg_recall_lesion = 0
        
        
        self.sum_dice_liver = 0
        self.sum_precision_liver = 0
        self.sum_recall_liver = 0
        
        self.sum_dice_lesion = 0
        self.sum_precision_lesion = 0
        self.sum_recall_lesion = 0
        
        self.count = 0

    def update(self,label_preds, label_trues):
        
        self.val_dice_liver, self.val_precision_liver, self.val_recall_liver, self.val_dice_lesion,
        self.val_precision_lesion, self.val_recall_lesion, n = self.eval_metrics(label_preds, label_trues)

        self.sum_dice_liver += self.val_dice_liver * n
        self.sum_precision_liver += self.val_precision_liver * n
        self.sum_recall_liver += self.val_recall_liver * n
        
        self.sum_dice_lesion += self.val_dice_lesion * n
        self.sum_precision_lesion += self.val_precision_lesion * n
        self.sum_recall_lesion += self.val_recall_lesion * n
        
        self.count += n
        
        self.avg_dice_liver = self.sum_dice_liver / self.count
        self.avg_precision_liver = self.sum_precision_liver / self.count
        self.avg_recall_liver = self.sum_recall_liver / self.count
        
        self.avg_dice_lesion = self.sum_dice_lesion / self.count
        self.avg_precision_lesion = self.sum_precision_lesion / self.count
        self.avg_recall_lesion = self.sum_recall_lesion / self.count
        
        
    def eval_metrics(self, label_preds, label_trues):
        """Inputs:
            - label_preds: BxCxHxW
            - label_trues: BxHxW
        
        
        """
        np_preds = label_preds.cpu().numpy()
        np_trues = label_trues.cpu().numpy()
        
        batch_size = np_trues.shape[0]
        
        pred_mask = np.zeros_like(np_preds, dtype=int)
        pred_mask[np_preds.max(axis=1,keepdims=1) == np_preds] = 1

        true_liver = np.where(np_trues == 1, 1, 0)
        true_lesion = np.where(np_trues == 2, 1, 0)

        pred_liver = pred_mask[:, 1, :, :]
        pred_lesion = pred_mask[:, 2, :, :]

        tp_liver = np.sum(pred_liver*true_liver, axis=(1,2))
        tp_lesion = np.sum(pred_lesion*true_lesion, axis=(1,2))
        
        fp_liver = np.sum(pred_liver, axis=(1,2)) - tp_liver
        fp_lesion = np.sum(pred_lesion, axis=(1,2)) - tp_lesion
        
        fn_liver = np.sum(true_liver, axis=(1,2)) - tp_liver
        fn_lesion = np.sum(true_lesion, axis=(1,2)) - tp_lesion
        
        dice_liver = np.average((2*tp_liver + 1)/(2*tp_liver + 1 + fp_liver + fn_liver))
        dice_lesion = np.average((2*tp_lesion + 1)/(2*tp_lesion + 1 + fp_lesion + fn_lesion))
        
        precision_liver = np.average((tp_liver + 1)/(tp_liver + 1 + fp_liver))
        precision_lesion = np.average((tp_lesion + 1)/(tp_lesion + 1 + fp_lesion))
        
        recall_liver = np.average((tp_liver + 1)/(tp_liver + 1 + fp_liver))
        recall_lesion = np.average((tp_lesion + 1)/(tp_lesion + 1 + fn_lesion))
        
        return np.array([dice_liver, precision_liver, recall_liver, dice_lesion, precision_lesion, recall_lesion, batch_size])
    
    def get_scores(self):
        return np.array([self.avg_dice_liver,
        self.avg_precision_liver,
        self.avg_recall_liver,
        self.avg_dice_lesion,
        self.avg_precision_lesion,
        self.avg_recall_lesion])
