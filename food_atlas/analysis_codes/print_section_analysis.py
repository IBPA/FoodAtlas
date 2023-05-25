from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
)
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection


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


if __name__ == '__main__':
    # Load validation result data.
    data_val_prob_dfs = []
    for seed in range(50):
        data_val_dfs = []
        for fold in range(10):
            data_val_fold = pd.read_csv(
                f"/data/lfz/projects/FoodAtlas/outputs/entailment_model/"
                f"10_folds_analysis/fold_{fold}/seed_{seed}/predicted.tsv",
                sep='\t')
            data_val_dfs += [data_val_fold]
        data_val = pd.concat(data_val_dfs, ignore_index=True)

        data_val_prob = data_val[['prob']].rename(
            columns={'prob': f'prob_{seed}'}
        )
        data_val_prob_dfs += [data_val_prob]

    data_val_probs = pd.concat(data_val_prob_dfs, axis=1)
    data_val_probs['mean'] = data_val_probs.mean(axis=1)
    data_val_probs['std'] = data_val_probs.std(axis=1)
    data = pd.concat(
        [data_val,
         data_val_probs],
        axis=1,
    )
    print(data['section'].value_counts(dropna=False))

    data['pred'] = data['mean'].apply(
        lambda x: 'Entails' if x > 0.5 else 'Does not entail')
    data['answer'] = data['answer'].replace(
        {'Entails': 1, 'Does not entail': 0}
    )
    data['pred'] = data['pred'].replace(
        {'Entails': 1, 'Does not entail': 0}
    )
    data['section'] = data['section'].fillna('Others')

    # Get all metrics for each section.
    for sec in data['section'].unique().tolist():
        data_ = data.query(f"section == '{sec}'")
        try:
            result = get_all_metrics(
                data_['answer'], data_['pred'], data_['mean'])
            print(f"{sec}: {result['precision']}")
        except Exception:
            pass

    # # Compare each pair of sections to see if they are significantly
    # #   different.
    # sec_1_list = []
    # sec_2_list = []
    # ps = []
    # for sec_1, sec_2 in combinations(data['section'].unique().tolist(), 2):
    #     data_1 = data.query(f"section == '{sec_1}'")
    #     data_2 = data.query(f"section == '{sec_2}'")
    #     try:
    #         _, p = ttest_ind(data_1['mean'], data_2['mean'])
    #         sec_1_list += [sec_1]
    #         sec_2_list += [sec_2]
    #         ps += [p]
    #     except Exception:
    #         pass
    # ps_adj = fdrcorrection(
    #     ps,
    #     alpha=0.05,
    #     method='indep',
    # )[1]

    # for sec_1, sec_2, p_adj in zip(sec_1_list, sec_2_list, ps_adj):
    #     if p_adj < 0.05:
    #         print(f"{sec_1} vs {sec_2}: p_adj={p_adj}")

    # Check if precisions for the two groups are statistically different.
    #   Group 1: introduction, methods
    #   Group 2: abstract, title, conclusion
    group_1 = []
    group_2 = []
    for section in ['INTRO', 'METHODS', 'abstract', 'title', 'CONCL']:
        data_ = data.query(f"section == '{section}'")
        precs = []
        for seed in range(50):
            precs += [precision_score(
                data_['answer'],
                [1 if x > 0.5 else 0 for x in data_[f"prob_{seed}"].tolist()],
            )]
        print(f"{section}: precision = {np.mean(precs)} +/- {np.std(precs)}")

        if section in ['INTRO', 'METHODS']:
            group_1 += precs
        else:
            group_2 += precs

    print(ttest_ind(group_1, group_2))
