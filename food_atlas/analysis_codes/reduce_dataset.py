import pandas as pd

df_train_pool = pd.read_csv(
    "../../outputs/data_generation/train_pool.tsv",
    sep='\t',
    keep_default_na=False,
)

df_val = pd.read_csv(
    "../../outputs/data_generation/val.tsv",
    sep='\t',
    keep_default_na=False,
)

df_test = pd.read_csv(
    "../../outputs/data_generation/test.tsv",
    sep='\t',
    keep_default_na=False,
)

df_train_pool = df_train_pool.sample(n=1000, random_state=530)
df_val = df_val.sample(n=300, random_state=530)
df_test = df_test.sample(n=300, random_state=530)

df_train_pool.to_csv("../../outputs/data_generation/train_pool_small.tsv", sep='\t', index=False)
df_val.to_csv("../../outputs/data_generation/val_small.tsv", sep='\t', index=False)
df_test.to_csv("../../outputs/data_generation/test_small.tsv", sep='\t', index=False)
