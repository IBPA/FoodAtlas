from collections import Counter
import os
import random
import sys
import pandas as pd

scratch_dir = "/home/jasonyoun/Jason/Scratch/"
fa_dir = "/home/jasonyoun/Jason/Research/FoodAtlas/"

#
df_post_annotation_1 = pd.read_csv(
    scratch_dir + "data_generation/1/post_annotation_1.tsv", sep='\t'
)
df_post_annotation_2 = pd.read_csv(
    scratch_dir + "data_generation/2/random_sample_each_bin/post_annotation_2.tsv", sep='\t'
)
df_post_annotation_3 = pd.read_csv(
    scratch_dir + "data_generation/3/random_sample_each_bin/post_annotation_3.tsv", sep='\t'
)
df_post_annotation_4 = pd.read_csv(
    scratch_dir + "outputs/data_generation/1/post_annotation_1.tsv", sep='\t'
)
df_post_annotation_5 = pd.read_csv(
    scratch_dir + "outputs/data_generation/2/random_sample_each_bin/post_annotation_2.tsv", sep='\t'
)
df_post_annotation_6 = pd.read_csv(
    scratch_dir + "outputs/data_generation/3/random_sample_each_bin/post_annotation_3.tsv", sep='\t'
)
df_post_annotation = pd.concat([
    df_post_annotation_1,
    df_post_annotation_2,
    df_post_annotation_3,
    df_post_annotation_4,
    df_post_annotation_5,
    df_post_annotation_6,
])

#
df_val_post_annotation = pd.read_csv(
    scratch_dir + "outputs/data_generation/val_post_annotation.tsv",
    sep='\t',
)
df_test_post_annotation = pd.read_csv(
    scratch_dir + "outputs/data_generation/test_post_annotation.tsv",
    sep='\t',
)

# drop duplicates
# df_post_annotation = df_post_annotation.sample(frac=1)
rows, seen = [], []
for _, row in df_post_annotation.iterrows():
    hash_str = \
        str(row['pmid']) + str(row['pmcid']) + str(row['premise']) + \
        str(row['hypothesis_string']) + str(row['hypothesis_id'])
    seen.append(hash_str)
    if seen.count(hash_str) <= 2:
        rows.append(row)
df_post_annotation = pd.DataFrame(rows)

print("df_post_annotation.shape: ", df_post_annotation.shape)
print("df_val_post_annotation.shape: ", df_val_post_annotation.shape)
print("df_test_post_annotation.shape: ", df_test_post_annotation.shape)
print()

# #
# val_premises = list(set(df_val_post_annotation["premise"].tolist()))
# test_premises = list(set(df_test_post_annotation["premise"].tolist()))
# random.shuffle(val_premises)
# random.shuffle(test_premises)

# VAL_NUM_PREM = 0
# TEST_NUM_PREM = 0
# val_premises = val_premises[VAL_NUM_PREM:]
# test_premises = test_premises[TEST_NUM_PREM:]
# additional_train_premises = val_premises[0:VAL_NUM_PREM] + test_premises[0:TEST_NUM_PREM]

# # clean up val/test
# rows = []
# rows_train = []
# for _, row in df_val_post_annotation.iterrows():
#     if row["premise"] in val_premises:
#         rows.append(row)
#     else:
#         rows_train.append(row)
# df_val_post_annotation = pd.DataFrame(rows)

# rows = []
# for _, row in df_test_post_annotation.iterrows():
#     if row["premise"] in test_premises:
#         rows.append(row)
#     else:
#         rows_train.append(row)
# df_test_post_annotation = pd.DataFrame(rows)

# df_post_annotation = pd.concat([df_post_annotation, pd.DataFrame(rows_train)])

# print("df_post_annotation.shape: ", df_post_annotation.shape)
# print("df_val_post_annotation.shape: ", df_val_post_annotation.shape)
# print("df_test_post_annotation.shape: ", df_test_post_annotation.shape)
# print()

# drop overlapping hypothesis
val_hyp = list(set(df_val_post_annotation["hypothesis_string"].tolist()))
test_hyp = list(set(df_test_post_annotation["hypothesis_string"].tolist()))
val_test_hyp = list(set(val_hyp + test_hyp))

rows = []
for _, row in df_post_annotation.iterrows():
    if row["hypothesis_string"] not in val_test_hyp:
        rows.append(row)
df_post_annotation = pd.DataFrame(rows)

rows = []
for _, row in df_test_post_annotation.iterrows():
    if row["hypothesis_string"] not in val_hyp:
        rows.append(row)
df_test_post_annotation = pd.DataFrame(rows)

print("df_post_annotation.shape: ", df_post_annotation.shape)
print("df_val_post_annotation.shape: ", df_val_post_annotation.shape)
print("df_test_post_annotation.shape: ", df_test_post_annotation.shape)
print()

# train
pre_annotation_columns = [
    "pmid",
    "pmcid",
    "section",
    "premise",
    "head",
    "relation",
    "tail",
    "hypothesis_string",
    "hypothesis_id",
    "source",
]

df_post_annotation["source"] = "annotation:train"
df_post_annotation.to_csv(
    fa_dir + "outputs/data_generation/train_pool_post_annotation.tsv",
    sep='\t',
    index=False,
)

df_train_pre_annotation = df_post_annotation[pre_annotation_columns].copy()
df_train_pre_annotation.drop_duplicates(
    subset=["premise", "hypothesis_string", "hypothesis_id"], inplace=True)
df_train_pre_annotation["source"] = "annotation:train"
df_train_pre_annotation.to_csv(
    fa_dir + "outputs/data_generation/train_pool_pre_annotation.tsv",
    sep='\t',
    index=False,
)

# val
pre_annotation_columns = [
    "pmid",
    "pmcid",
    "section",
    "premise",
    "head",
    "relation",
    "tail",
    "hypothesis_string",
    "hypothesis_id",
]

df_val_post_annotation["source"] = "annotation:val"
df_val_post_annotation.to_csv(
    fa_dir + "outputs/data_generation/val_post_annotation.tsv",
    sep='\t',
    index=False,
)

df_val_pre_annotation = df_val_post_annotation[pre_annotation_columns].copy()
df_val_pre_annotation.drop_duplicates(
    subset=["premise", "hypothesis_string", "hypothesis_id"], inplace=True)
df_val_pre_annotation["source"] = "annotation:val"
df_val_pre_annotation.to_csv(
    fa_dir + "outputs/data_generation/val_pre_annotation.tsv",
    sep='\t',
    index=False,
)

# test
df_test_post_annotation["source"] = "annotation:test"
df_test_post_annotation.to_csv(
    fa_dir + "outputs/data_generation/test_post_annotation.tsv",
    sep='\t',
    index=False,
)

df_test_pre_annotation = df_test_post_annotation[pre_annotation_columns].copy()
df_test_pre_annotation.drop_duplicates(
    subset=["premise", "hypothesis_string", "hypothesis_id"], inplace=True)
df_test_pre_annotation["source"] = "annotation:test"
df_test_pre_annotation.to_csv(
    fa_dir + "outputs/data_generation/test_pre_annotation.tsv",
    sep='\t',
    index=False,
)
