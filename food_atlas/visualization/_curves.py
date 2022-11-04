from sklearn.metrics import (
    precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_test_prob_result(
        run_ids,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain']):
    PATH_REUSLT = "~/git/FoodAtlas/outputs/data_generation/{}/run_{}/"\
        "round_{}/test_probs.tsv"

    result_rows = []
    for al in active_learning_strategies:
        for run_id in run_ids:
            result = pd.read_csv(
                PATH_REUSLT.format(al, run_id, 10),
                sep='\t')

            result_rows += [{
                'al': al,
                'run_id': run_id,
                'labels': result['answer'].values,
                'probs': result['prob'].values,
            }]

    result = pd.DataFrame(result_rows)

    return result


def plot_curves_all(
        run_ids,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain'],
        path_save=None):
    """
    """
    result = get_test_prob_result(
        run_ids,
        active_learning_strategies,
    )

    ALPHA = 0.1
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i, al in enumerate(active_learning_strategies):
        result_al = result[result['al'] == al]

        aps = []
        aurocs = []
        with tqdm(total=len(result_al)) as pbar:
            for ir, row in enumerate(result_al.itertuples()):
                labels = [1 if x == 'Entails' else 0 for x in row.labels]
                prec, rec, _ = precision_recall_curve(labels, row.probs)
                fpr, tpr, _ = roc_curve(labels, row.probs)
                aps += [average_precision_score(labels, row.probs)]
                aurocs += [roc_auc_score(labels, row.probs)]

                sns.lineplot(
                    x=rec,
                    y=prec,
                    color='#e6e6e6',
                    alpha=ALPHA,
                    label=al if ir == 0 else None,
                    ax=axs[0],
                )
                sns.lineplot(
                    x=fpr,
                    y=tpr,
                    hue_order=active_learning_strategies,
                    color='#e6e6e6',
                    alpha=ALPHA,
                    label=al if ir == 0 else None,
                    ax=axs[1],
                )
                pbar.update(1)

        print(
            f"{al}\n"
            f"    average precision: {np.array(aps).mean()} +/- "
            f"{np.array(aps).std()}\n"
            f"    auroc            : {np.array(aurocs).mean()} +/- "
            f"{np.array(aurocs).std()}"
        )

    axs[0].set_title('Precision-Recall Curve')
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')

    axs[1].set_title('ROC Curve')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()


def plot_curves_average(
        run_ids,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain'],
        path_save=None):
    """
    """
    result = get_test_prob_result(
        run_ids,
        active_learning_strategies,
    )

    ALPHA = 0.1
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i, al in enumerate(active_learning_strategies):
        result_al = result[result['al'] == al]

        labels_total = []
        probs_total = []
        for row in result_al.itertuples():
            labels_total += row.labels.tolist()
            probs_total += row.probs.tolist()

        labels_total = [1 if x == 'Entails' else 0 for x in labels_total]
        prec, rec, _ = precision_recall_curve(labels_total, probs_total)
        fpr, tpr, _ = roc_curve(labels_total, probs_total)
        ap = average_precision_score(labels_total, probs_total)
        auroc = roc_auc_score(labels_total, probs_total)

        sns.lineplot(
            x=rec,
            y=prec,
            color='#e6e6e6',
            label=al,
            ax=axs[0],
        )
        sns.lineplot(
            x=fpr,
            y=tpr,
            color='#e6e6e6',
            label=al,
            ax=axs[1],
        )
        print(
            f"{al}\n"
            f"    average precision: {ap}\n"
            f"    auroc            : {auroc}"
        )

    axs[0].set_title('Precision-Recall Curve')
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')

    axs[1].set_title('ROC Curve')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()
