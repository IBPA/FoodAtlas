import pandas as pd


def _bin(df):
    print(f"Original: {df.shape[0]}")
    df_subset = df[df["prob"] <= 0.1]
    print(f"0.0 <= prob <= 0.1: {df_subset.shape[0]}")
    df_subset = df[(df["prob"] > 0.1) & (df["prob"] <= 0.2)]
    print(f"0.1 < prob <= 0.2: {df_subset.shape[0]}")
    df_subset = df[(df["prob"] > 0.2) & (df["prob"] <= 0.3)]
    print(f"0.2 < prob <= 0.3: {df_subset.shape[0]}")
    df_subset = df[(df["prob"] > 0.3) & (df["prob"] <= 0.4)]
    print(f"0.3 < prob <= 0.4: {df_subset.shape[0]}")
    df_subset = df[(df["prob"] > 0.4) & (df["prob"] <= 0.5)]
    print(f"0.4 < prob <= 0.5: {df_subset.shape[0]}")
    df_subset = df[(df["prob"] > 0.5) & (df["prob"] <= 0.6)]
    print(f"0.5 < prob <= 0.6: {df_subset.shape[0]}")
    df_subset = df[(df["prob"] > 0.6) & (df["prob"] <= 0.7)]
    print(f"0.6 < prob <= 0.7: {df_subset.shape[0]}")
    df_subset = df[(df["prob"] > 0.7) & (df["prob"] <= 0.8)]
    print(f"0.7 < prob <= 0.8: {df_subset.shape[0]}")
    df_subset = df[(df["prob"] > 0.8) & (df["prob"] <= 0.9)]
    print(f"0.8 < prob <= 0.9: {df_subset.shape[0]}")
    df_subset = df[df["prob"] > 0.9]
    print(f"0.9 < prob: {df_subset.shape[0]}")


print("R1 - predicted")
df_r1 = pd.read_csv(
    "../../outputs/data_generation/1/predicted_1.tsv",
    sep='\t',
    keep_default_na=False,
)
r1_result = _bin(df_r1)

print()
print("R2 - predicted (random)")
df_r2 = pd.read_csv(
    "../../outputs/data_generation/2/random_sample_each_bin/predicted_2.tsv",
    sep='\t',
    keep_default_na=False,
)
r2_result = _bin(df_r2)
