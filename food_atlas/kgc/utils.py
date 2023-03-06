import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def determine_threshold(scores, gt_label):
    f1_scores = []
    thresholds = []

    for score in sorted(scores):
        predictions = [1 if x >= score else 0 for x in scores]
        thresholds.append(score)
        f1_scores.append(f1_score(gt_label, predictions))

    idx = np.argmax(f1_scores)
    return thresholds[idx]
