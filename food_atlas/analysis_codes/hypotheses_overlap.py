import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

train_files = [
    "../../outputs/data_generation/1/train_1.tsv",
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
    "/home/jasonyoun/Jason/Scratch/temp/test_1_with_score.tsv",
    sep='\t',
    keep_default_na=False,
)
test_hypotheses = list(set(df_test["hypothesis_string"].tolist()))

train_overlap = [x for x in train_hypotheses if x in test_hypotheses]
print(len(train_overlap))

overlapping = []
non_overlapping = []
for idx, row in df_test.iterrows():
    if row["hypothesis_string"] in train_hypotheses:
        overlapping.append(row["prob"])
    else:
        non_overlapping.append(row["prob"])

print(len(overlapping))
print(len(non_overlapping))

print(np.mean(overlapping))
print(np.mean(non_overlapping))

print(ttest_ind(overlapping, non_overlapping, equal_var=False))
