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

    data_mean_score = pd.concat(
        [
            pd.read_csv(
                f"outputs/data_generation/certain_pos/run_1/round_10/"
                f"test_probs.tsv",
                sep='\t',
            ),
            data['mean'],
        ],
        axis=1,
    ).drop(columns=['answer', 'prob'])

    data_mean_score.to_csv(
        "outputs/data_processing/predictions_test.tsv",
        sep='\t',
        index=False,
    )
