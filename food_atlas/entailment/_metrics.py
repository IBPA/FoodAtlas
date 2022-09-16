from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


def get_all_metrics(y_true, y_pred, y_score=None):
    """
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'f1': 2 * tp / (2 * tp + fp + fn),
        'auroc': roc_auc_score(
            y_true, y_score) if y_score is not None else None,
        'ap': average_precision_score(
            y_true, y_score) if y_score is not None else None,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }

    return metrics
