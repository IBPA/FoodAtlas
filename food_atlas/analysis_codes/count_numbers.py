from collections import Counter
import os
import sys
import pandas as pd

sys.path.append('..')
from common_utils.foodatlas_types import CandidateEntity, CandidateRelation  # noqa: E402
from common_utils.foodatlas_types import FoodAtlasEntity, FoodAtlasRelation  # noqa: E402

root_dir = "/home/jasonyoun/Jason/Research/FoodAtlas"

# count number of query food names
df_food_names = pd.read_csv(
    os.path.join(root_dir, 'data/FooDB/foodb_foods.txt'),
    sep='\t',
    keep_default_na=False,
)

names = list(set(df_food_names["name"].tolist()))
scientific_names = list(set(df_food_names["name_scientific"].tolist()))

query_foods = [x for x in names if x != ""] + [x for x in scientific_names if x != ""]
print(f"Number of query foods: {len(query_foods)}")

# count PH pairs
df_ph_pairs = pd.read_csv(
    os.path.join(root_dir, 'outputs/data_generation/ph_pairs_20220721_100213.txt'),
    sep='\t',
    keep_default_na=False,
)
print(df_ph_pairs)
print(df_ph_pairs.columns)

pmids = list(set(df_ph_pairs["pmid"].tolist()))
print(f"Number of PMIDs: {len(pmids)}")

premises = list(set(df_ph_pairs["premise"].tolist()))
print(f"Number of premises: {len(premises)}")

# count possible entities
df_ph_pairs["head"] = df_ph_pairs["head"].apply(lambda x: eval(x, globals()))
df_ph_pairs["relation"] = df_ph_pairs["relation"].apply(lambda x: eval(x, globals()))
df_ph_pairs["tail"] = df_ph_pairs["tail"].apply(lambda x: eval(x, globals()))

print(df_ph_pairs)

fa_ent = FoodAtlasEntity('')

added = []

for idx, row in df_ph_pairs.iterrows():
    print(f"{idx+1}/{df_ph_pairs.shape[0]}")

    head = row.at["head"]
    tail = row.at["tail"]
    hypothesis_id = row.at["hypothesis_id"]

    # head
    if head.type == "species_with_part":
        other_db_ids = dict([hypothesis_id.split('-')[0].split(':')])
    else:
        other_db_ids = head.other_db_ids

    if str(head) not in added:
        fa_ent.add(
            type_=head.type,
            name=head.name,
            synonyms=head.synonyms,
            other_db_ids=other_db_ids,
        )
        added.append(str(head))

    # tail
    if tail.type == "species_with_part":
        other_db_ids = dict([hypothesis_id.split('-')[0].split(':')])
    else:
        other_db_ids = tail.other_db_ids

    if str(tail) not in added:
        fa_ent.add(
            type_=tail.type,
            name=tail.name,
            synonyms=tail.synonyms,
            other_db_ids=other_db_ids,
        )
        added.append(str(tail))

fa_ent.save(os.path.join(root_dir, 'outputs/analysis_codes/entities.txt'))

# count val/test stats
df_val = pd.read_csv(
    os.path.join(root_dir, "outputs/data_generation/val.tsv"),
    sep='\t',
)

df_test = pd.read_csv(
    os.path.join(root_dir, "outputs/data_generation/test.tsv"),
    sep='\t',
)

val_answer = df_val.answer.tolist()
test_answer = df_test.answer.tolist()

print(Counter(val_answer))
print(Counter(test_answer))
