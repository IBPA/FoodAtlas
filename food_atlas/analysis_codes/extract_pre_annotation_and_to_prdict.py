import pandas as pd

df_post_annotation = pd.read_csv(
    "../../outputs/data_generation/3/random_sample_each_bin/post_annotation_3.tsv",
    sep='\t',
    keep_default_na=False,
)

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
df_pre_annotation = df_post_annotation[pre_annotation_columns].copy()
df_pre_annotation.drop_duplicates(inplace=True)
df_pre_annotation.to_csv(
    "../../outputs/data_generation/3/random_sample_each_bin/pre_annotation_3.tsv",
    sep='\t',
    index=False
)

#
df_predicted = pd.read_csv(
    "../../outputs/data_generation/2/random_sample_each_bin/predicted_2.tsv",  # change this later!!!
    sep='\t',
    keep_default_na=False,
)
print("df_predicted:\n", df_predicted)
print()

#
hash_str_lookup = []
for _, row in df_pre_annotation.iterrows():
    hash_str_lookup.append(
        str(row['pmid']) + str(row['pmcid']) + str(row['section']) + str(row['premise']) +
        str(row['head']) + str(row['relation']) + str(row['tail']) +
        str(row['hypothesis_string']) + str(row['hypothesis_id'])
    )

new_rows = []
for _, row in df_predicted.iterrows():
    hash_str = \
        str(row['pmid']) + str(row['pmcid']) + str(row['section']) + str(row['premise']) + \
        str(row['head']) + str(row['relation']) + str(row['tail']) + \
        str(row['hypothesis_string']) + str(row['hypothesis_id'])

    if hash_str not in hash_str_lookup:
        new_rows.append(row)

df_next_to_predict = pd.DataFrame(new_rows)
next_to_predict_columns = [
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
df_next_to_predict = df_next_to_predict[next_to_predict_columns]
df_next_to_predict.to_csv(
    "../../outputs/data_generation/3/random_sample_each_bin/to_predict_3.tsv",
    sep='\t',
    index=False,
)
