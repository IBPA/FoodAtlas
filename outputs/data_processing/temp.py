import sys
from copy import deepcopy
sys.path.append('../../food_atlas/data_processing/')
import pandas as pd
from common_utils.utils import read_dataframe

file_names = [
    "folds_for_prod_model/fold_0/train.tsv",
    "folds_for_prod_model/fold_0/val.tsv",
    "folds_for_prod_model/fold_1/train.tsv",
    "folds_for_prod_model/fold_1/val.tsv",
    "folds_for_prod_model/fold_2/train.tsv",
    "folds_for_prod_model/fold_2/val.tsv",
    "folds_for_prod_model/fold_3/train.tsv",
    "folds_for_prod_model/fold_3/val.tsv",
    "folds_for_prod_model/fold_4/train.tsv",
    "folds_for_prod_model/fold_4/val.tsv",
    "folds_for_prod_model/fold_5/train.tsv",
    "folds_for_prod_model/fold_5/val.tsv",
    "folds_for_prod_model/fold_6/train.tsv",
    "folds_for_prod_model/fold_6/val.tsv",
    "folds_for_prod_model/fold_7/train.tsv",
    "folds_for_prod_model/fold_7/val.tsv",
    "folds_for_prod_model/fold_8/train.tsv",
    "folds_for_prod_model/fold_8/val.tsv",
    "folds_for_prod_model/fold_9/train.tsv",
    "folds_for_prod_model/fold_9/val.tsv",
]

df_parts = pd.read_csv('../../data/FoodAtlas/food_parts.txt', sep='\t')
parts_lookup = dict(zip(
    df_parts["food_part"].tolist(),
    df_parts["foodatlas_part_id"].tolist()
))


for file_name in file_names:
    df = read_dataframe(file_name)

    data = []
    for idx, row in df.iterrows():
        head = row["head"]
        if head.type == "organism":
            new_other_db_ids = deepcopy(head.other_db_ids)
            new_other_db_ids["foodatlas_part_id"] = "p0"
            head = head._replace(other_db_ids=new_other_db_ids)
        elif head.type == "organism_with_part":
            new_other_db_ids = deepcopy(head.other_db_ids)
            part_name = head.name.split(' - ')[1]
            new_other_db_ids["foodatlas_part_id"] = parts_lookup[part_name]
            head = head._replace(other_db_ids=new_other_db_ids)

        row["head"] = head
        data.append(deepcopy(row))

    df_new = pd.DataFrame(data)
    df_new.to_csv(file_name, sep='\t', index=False)
