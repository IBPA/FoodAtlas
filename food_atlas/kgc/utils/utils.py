import sys

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def determine_threshold(scores, gt_label):
    f1_scores = []
    thresholds = []

    for score in sorted(scores):
        predictions = [1 if x >= score else 0 for x in scores]
        thresholds.append(score)
        f1_scores.append(f1_score(gt_label, predictions))

    idx = np.argmax(f1_scores)
    return thresholds[idx]


def generate_report(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'precision_score': precision_score(y_true, y_pred),
        'recall_score': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'accuracy_score': accuracy_score(y_true, y_pred),
        'average_precision_score': average_precision_score(y_true, y_score) if y_score else None,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'roc_auc_score': roc_auc_score(y_true, y_score) if y_score else None,
    }
