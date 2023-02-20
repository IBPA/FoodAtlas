import math

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


def map_bins(p):
    if p < 0.1:
        return 0
    elif p < 0.2:
        return 1
    elif p < 0.3:
        return 2
    elif p < 0.4:
        return 3
    elif p < 0.5:
        return 4
    elif p < 0.6:
        return 5
    elif p < 0.7:
        return 6
    elif p < 0.8:
        return 7
    elif p < 0.9:
        return 8
    else:
        return 9


def hypergeometric_test(
        N: int,
        K: int,
        n: int,
        k: int) -> float:
    """Compute the hypergeometric test.

    Args:
        N (int): Population size.
        K (int): Number of draws.
        n (int): Number of successes in the population.
        k (int): Number of successes in the sample.

    Returns:
        float: The p-value.
    """
    return math.comb(n, k) * math.comb(N - n, K - k) / math.comb(N, K)


if __name__ == '__main__':

    train_files = [
        "/data/lfz/projects/FoodAtlas/outputs/entailment_model_overlap_backup/1/train_1_merged.tsv",
        # "../../outputs/data_generation/2/random_sample_each_bin/train_2.tsv",
        # "../../outputs/data_generation/3/random_sample_each_bin/train_3.tsv",
    ]
    data = []
    for x in train_files:
        df = pd.read_csv(x, sep='\t', keep_default_na=False)
        data.append(df)

    df_train = pd.concat(data)
    train_hypotheses = list(set(df_train["hypothesis_string"].tolist()))
    df_test = pd.read_csv(
        "outputs/test_1_with_score.tsv",
        sep='\t',
        keep_default_na=False,
    )
    test_hypotheses = list(set(df_test["hypothesis_string"].tolist()))
    train_overlap = [x for x in train_hypotheses if x in test_hypotheses]
    overlapping = []
    non_overlapping = []
    for idx, row in df_test.iterrows():
        if row["hypothesis_string"] in train_hypotheses:
            overlapping.append(row["prob"])
        else:
            non_overlapping.append(row["prob"])

    data_plot = pd.DataFrame([
        overlapping + non_overlapping,
        ['overlapping'] * len(overlapping)
        + ['non-overlapping'] * len(non_overlapping)],
        index=['prob', 'overlap']).T
    data_plot['bin'] = data_plot['prob'].apply(map_bins)
    print(data_plot)
    sns.histplot(
        data=data_plot,
        x='prob',
        hue='overlap',
        bins=10,
    )
    plt.savefig(f"outputs/overlapping.png", bbox_inches='tight')
    plt.close()

    for bin in range(10):
        data_bin = data_plot[data_plot['bin'] == bin]
        N = len(data_plot)
        K = len(data_bin)
        n = len(non_overlapping)
        k = len(data_bin[data_bin['overlap'] == 'non-overlapping'])
        p = hypergeometric_test(N, K, n, k)
        print(f"bin {bin}: {p}")
