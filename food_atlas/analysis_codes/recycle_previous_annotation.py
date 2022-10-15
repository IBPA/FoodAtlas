import sys
import pandas as pd

scratch_outputs_dir = "/home/jasonyoun/Jason/Scratch/outputs/"

df_prev_pre_annotation_2 = pd.read_csv(
    scratch_outputs_dir + "data_generation/2/random_sample_each_bin/pre_annotation_2.tsv",
    sep='\t',
    keep_default_na=False,
)
df_prev_pre_annotation_3 = pd.read_csv(
    scratch_outputs_dir + "data_generation/3/random_sample_each_bin/pre_annotation_3.tsv",
    sep='\t',
    keep_default_na=False,
)
df_prev_pre_annotation = pd.concat([df_prev_pre_annotation_2, df_prev_pre_annotation_3])
df_prev_pre_annotation.drop("source", axis=1, inplace=True)
df_prev_pre_annotation["prob"] = ""
print("df_prev_pre_annotation.columns:\n", df_prev_pre_annotation.columns)
print()

df_prev_post_annotation_2 = pd.read_csv(
    scratch_outputs_dir + "data_generation/2/random_sample_each_bin/post_annotation_2.tsv",
    sep='\t',
    keep_default_na=False,
)
df_prev_post_annotation_3 = pd.read_csv(
    scratch_outputs_dir + "data_generation/3/random_sample_each_bin/post_annotation_3.tsv",
    sep='\t',
    keep_default_na=False,
)
df_prev_post_annotation = pd.concat([df_prev_post_annotation_2, df_prev_post_annotation_3])
print("df_prev_post_annotation.columns:\n", df_prev_post_annotation.columns)
print()

df_predicted = pd.read_csv(
    "../../outputs/data_generation/1/predicted_1.tsv",  # change this later!!!
    sep='\t',
    keep_default_na=False,
)
print("df_predicted.columns:\n", df_predicted.columns)
print()

df_concat = pd.concat([df_predicted, df_prev_pre_annotation])
subset = list(df_concat.columns)
subset.remove("prob")
print("subset: ", subset)
df_overlap = df_concat[df_concat.duplicated(subset=subset, keep='last')]

# recyle now
df_overlap_dict = {
    "df_overlap_0_10": df_overlap[df_overlap["prob"] <= 0.1],
    "df_overlap_10_20": df_overlap[(df_overlap["prob"] > 0.1) & (df_overlap["prob"] <= 0.2)],
    "df_overlap_20_30": df_overlap[(df_overlap["prob"] > 0.2) & (df_overlap["prob"] <= 0.3)],
    "df_overlap_30_40": df_overlap[(df_overlap["prob"] > 0.3) & (df_overlap["prob"] <= 0.4)],
    "df_overlap_40_50": df_overlap[(df_overlap["prob"] > 0.4) & (df_overlap["prob"] <= 0.5)],
    "df_overlap_50_60": df_overlap[(df_overlap["prob"] > 0.5) & (df_overlap["prob"] <= 0.6)],
    "df_overlap_60_70": df_overlap[(df_overlap["prob"] > 0.6) & (df_overlap["prob"] <= 0.7)],
    "df_overlap_70_80": df_overlap[(df_overlap["prob"] > 0.7) & (df_overlap["prob"] <= 0.8)],
    "df_overlap_80_90": df_overlap[(df_overlap["prob"] > 0.8) & (df_overlap["prob"] <= 0.9)],
    "df_overlap_90_100": df_overlap[(df_overlap["prob"] > 0.9) & (df_overlap["prob"] <= 1.0)],
}

df_to_concat = []
to_annotate_num_dict = {}
for key, df in df_overlap_dict.items():
    print(key, ': ', df.shape[0])

    if df.shape[0] >= 100:
        df_to_concat.append(df.sample(n=100))
        to_annotate_num_dict[key] = 0
    else:
        df_to_concat.append(df)
        to_annotate_num_dict[key] = 100 - df.shape[0]
print()

df_prev_pre_annotation_recyclable = pd.concat(df_to_concat)
print("df_prev_pre_annotation_recyclable:\n", df_prev_pre_annotation_recyclable)
print("df_prev_pre_annotation_recyclable.columns:\n", df_prev_pre_annotation_recyclable.columns)
print("to_annotate_num_dict: ", to_annotate_num_dict)

#
recyclable_hash_str = []
for _, row in df_prev_pre_annotation_recyclable.iterrows():
    recyclable_hash_str.append(
        str(row['pmid']) + str(row['pmcid']) + str(row['section']) + str(row['premise']) +
        str(row['head']) + str(row['relation']) + str(row['tail']) +
        str(row['hypothesis_string']) + str(row['hypothesis_id'])
    )

new_rows = []
for _, row in df_prev_post_annotation.iterrows():
    hash_str = \
        str(row['pmid']) + str(row['pmcid']) + str(row['section']) + str(row['premise']) + \
        str(row['head']) + str(row['relation']) + str(row['tail']) + \
        str(row['hypothesis_string']) + str(row['hypothesis_id'])

    if hash_str in recyclable_hash_str:
        new_rows.append(row)

df_new_post_annotation = pd.DataFrame(new_rows)
df_new_post_annotation.to_csv(
    "/home/jasonyoun/Jason/Scratch/temp/new_post_annotation.tsv",
    sep='\t',
    index=False,
)

#
new_data_list = []
for key, val in to_annotate_num_dict.items():
    if val == 0:
        continue

    print(key, val)

    lower_range = int(key.split('_')[-2])/100
    upper_range = int(key.split('_')[-1])/100

    df_predicted_subset = \
        df_predicted[(df_predicted["prob"] > lower_range) &
                     (df_predicted["prob"] <= upper_range)]

    condition = True
    while condition:
        df_newly_sampled = df_predicted_subset.sample(n=val)
        conditions = []
        for _, row in df_newly_sampled.iterrows():
            hash_str = \
                str(row['pmid']) + str(row['pmcid']) + str(row['section']) + str(row['premise']) + \
                str(row['head']) + str(row['relation']) + str(row['tail']) + \
                str(row['hypothesis_string']) + str(row['hypothesis_id'])

            if hash_str in recyclable_hash_str:
                conditions.append(True)
            else:
                conditions.append(False)
        condition = True in conditions

    new_data_list.append(df_newly_sampled)

df_to_annotate = pd.concat(new_data_list)
df_to_annotate.drop("prob", axis=1, inplace=True)
df_to_annotate["source"] = "annotation:round_2"
df_to_annotate.to_csv(
    "/home/jasonyoun/Jason/Scratch/temp/new_r2_to_annotate.tsv",
    sep='\t',
    index=False,
)
