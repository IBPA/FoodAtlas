from itertools import product

from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
import pandas as pd

from tqdm import tqdm


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
        'npv': tn / (tn + fn),
        'pp': tp + fp,
        'pn': tn + fn,
    }

    return metrics


if __name__ == '__main__':
    # data_cols = []
    # for al, run_id in tqdm(product(
    #             ['certain_pos', 'random', 'stratified', 'uncertain'],
    #             range(1, 1 + 100)
    #         ),
    #         total=4 * 100,
    #         ):
    #     path_file = (
    #         f"outputs/data_generation/{al}/run_{run_id}/round_10/"
    #         f"test_probs.tsv"
    #     )
    #     data_file = pd.read_csv(path_file, sep='\t')
    #     data_cols += [data_file['prob']]

    # data = pd.concat(data_cols, axis=1)
    # data['mean'] = data.mean(axis=1)
    # data = pd.concat(
    #     [
    #         data_file[data_file.columns[:-1]],
    #         data['mean'],
    #     ],
    #     axis=1,
    # )

    # y_true = data['answer'].map({'Entails': 1, 'Does not entail': 0}).tolist()
    # y_pred = data['mean'].map(lambda x: 1 if x >= 0.5 else 0).tolist()
    # y_score = data['mean'].tolist()
    # print(get_all_metrics(y_true, y_pred, y_score=y_score))

    result = []
    for al, run_id in tqdm(product(
                ['certain_pos', 'random', 'stratified', 'uncertain'],
                range(1, 1 + 100)
            ),
            total=4 * 100,
            ):
        path_file = (
            f"outputs/data_generation/{al}/run_{run_id}/round_10/"
            f"test_probs.tsv"
        )
        data_file = pd.read_csv(path_file, sep='\t')

        y_true = data_file['answer'].map(
            {'Entails': 1, 'Does not entail': 0}).tolist()
        y_pred = data_file['prob'].map(
            lambda x: 1 if x >= 0.5 else 0).tolist()
        y_score = data_file['prob'].tolist()
        result += [get_all_metrics(y_true, y_pred, y_score=y_score)]

    result = pd.DataFrame(result)
    result = pd.concat(
        [
            result.mean().rename('mean'),
            result.std().rename('std'),
        ],
        axis=1,
    )
    print(result)
