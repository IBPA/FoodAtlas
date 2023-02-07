import os
import sys

sys.path.append('../data_processing/')

import pandas as pd  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

####################
# KG - annotations #
####################
KG_DIR = "../../outputs/kg/annotations"
fa_kg = KnowledgeGraph(
    kg_filepath=os.path.join(KG_DIR, "kg.txt"),
    evidence_filepath=os.path.join(KG_DIR, "evidence.txt"),
    entities_filepath=os.path.join(KG_DIR, "entities.txt"),
    relations_filepath=os.path.join(KG_DIR, "relations.txt"),
)

# entities
df_entities = fa_kg.get_all_entities()
print(f"Number of entities: {df_entities.shape[0]}")
print(f"Types of entities: {set(df_entities['type'].tolist())}")

df_chemicals = df_entities[df_entities["type"].apply(
    lambda x: x == "chemical" or x.startswith("chemical:"))]
print(f"Number of chemical entities: {df_chemicals.shape[0]}")

df_foods = df_entities[df_entities["type"].apply(
    lambda x: x == "organism" or x.startswith("organism:"))]
print(f"Number of food entities: {df_foods.shape[0]}")

df_food_with_part = df_entities[df_entities["type"].apply(
    lambda x: x == "organism_with_part" or x.startswith("organism_with_part:"))]
print(f"Number of food - part entities: {df_food_with_part.shape[0]}")

# relations
df_kg = fa_kg.get_kg()
print(f"Number of all triples: {df_kg.shape[0]}")

df_relations = fa_kg.get_all_relations()
relations_lookup = dict(zip(
    df_relations["name"].tolist(), df_relations["foodatlas_id"].tolist()))

df_hasPart = df_kg[df_kg["relation"] == relations_lookup["hasPart"]]
print(f"Number of hasPart triples: {df_hasPart.shape[0]}")

df_contains = df_kg[df_kg["relation"] == relations_lookup["contains"]]
print(f"Number of contains triples: {df_contains.shape[0]}")

################################
# KG - Annotations, MeSH, NCBI #
################################
KG_DIR = "../../outputs/kg/annotations_mesh_ncbi"
fa_kg = KnowledgeGraph(
    kg_filepath=os.path.join(KG_DIR, "kg.txt"),
    evidence_filepath=os.path.join(KG_DIR, "evidence.txt"),
    entities_filepath=os.path.join(KG_DIR, "entities.txt"),
    relations_filepath=os.path.join(KG_DIR, "relations.txt"),
)

# entities
df_entities = fa_kg.get_all_entities()
print(f"Number of entities: {df_entities.shape[0]}")
print(f"Types of entities: {set(df_entities['type'].tolist())}")

df_chemicals = df_entities[df_entities["type"].apply(
    lambda x: x == "chemical" or x.startswith("chemical:"))]
print(f"Number of chemical entities: {df_chemicals.shape[0]}")

df_foods = df_entities[df_entities["type"].apply(
    lambda x: x == "organism" or x.startswith("organism:"))]
print(f"Number of food entities: {df_foods.shape[0]}")

df_food_with_part = df_entities[df_entities["type"].apply(
    lambda x: x == "organism_with_part" or x.startswith("organism_with_part:"))]
print(f"Number of food - part entities: {df_food_with_part.shape[0]}")

# relations
df_kg = fa_kg.get_kg()
print(f"Number of all triples: {df_kg.shape[0]}")

df_relations = fa_kg.get_all_relations()
relations_lookup = dict(zip(
    df_relations["name"].tolist(), df_relations["foodatlas_id"].tolist()))

df_isA = df_kg[df_kg["relation"] == relations_lookup["isA"]]
print(f"Number of isA triples: {df_isA.shape[0]}")

df_hasChild = df_kg[df_kg["relation"] == relations_lookup["hasChild"]]
print(f"Number of hasChild triples: {df_hasChild.shape[0]}")

df_hasPart = df_kg[df_kg["relation"] == relations_lookup["hasPart"]]
print(f"Number of hasPart triples: {df_hasPart.shape[0]}")

df_contains = df_kg[df_kg["relation"] == relations_lookup["contains"]]
print(f"Number of contains triples: {df_contains.shape[0]}")


sys.exit()


df_evidence = fa_kg.get_evidence()
