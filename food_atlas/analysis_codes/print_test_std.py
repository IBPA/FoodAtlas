from itertools import product

import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm

if __name__ == '__main__':
    data_cols = []
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
        data_cols += [data_file['prob']]

    data = pd.concat(data_cols, axis=1)
    data['mean'] = data.mean(axis=1)
    data['std'] = data.std(axis=1)

    # Group 1: high confidence
    # Group 2: high uncertainty
    group_1 = data[(data['mean'] > 0.9) | (data['mean'] < 0.1)]
    group_2 = data[(data['mean'] > 0.4) & (data['mean'] < 0.6)]

    # Check if the two groups have different standard deviations.
    #   Result: pvalue=2.1750501832540284e-177
    print(group_1['std'].mean())  # 0.03899801010247441
    print(group_2['std'].mean())  # 0.3085325175196205
    print(ttest_ind(group_1['std'], group_2['std']))  # pvalue=2.175e-177
