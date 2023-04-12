from glob import glob
import os
from pathlib import Path
import sys

import pandas as pd

RESULTS_DIR = '../../../outputs/kgc/kg-bert'

root_folders = glob(os.path.join(RESULTS_DIR, '*'))
print(f'Number of root data folders: {len(root_folders)}')

hpo_folders = []
for x in root_folders:
    hpo_folders.extend(glob(os.path.join(x, 'hpo', '*')))
print(f'Number of HPO folders: {len(hpo_folders)}')

results = []
for x in hpo_folders:
    try:
        df = pd.read_csv(os.path.join(x, 'eval_link_prediction_metrics.tsv'), sep='\t')
    except OSError as e:
        print(f'Did not find files in {x}: {e}')
        continue
    df['model_path'] = x
    results.append(df)

df_results = pd.concat(results)
df_results.to_csv(os.path.join(RESULTS_DIR, 'hpo_results.txt'), sep='\t', index=False)
