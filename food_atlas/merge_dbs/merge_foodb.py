import argparse
import itertools
import os
import sys
import time
import xml.etree.ElementTree as ET

sys.path.append('..')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402
from common_utils.knowledge_graph import CandidateEntity, CandidateRelation  # noqa: E402
from common_utils.utils import save_pkl, load_pkl  # noqa: E402

FOODB_DATA_DIR = "../../data/FooDB/foodb_2020_04_07_csv"
COMPOUND_FILEPATH = os.path.join(FOODB_DATA_DIR, 'Compound.csv')
CONTENT_FILEPATH = os.path.join(FOODB_DATA_DIR, 'Content.csv')
FOOD_FILEPATH = os.path.join(FOODB_DATA_DIR, 'Food.csv')
KG_FILENAME = "kg.txt"
EVIDENCE_FILENAME = "evidence.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"

DATABASES_TO_INCLUDE = [
    'DTU',
    'USDA',
    'PHENOL EXPLORER',
    'KNAPSACK',
    'DUKE',
]


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        required=True,
        help="KG directory to merge the MESH to.",
    )

    parser.add_argument(
        "--output_kg_dir",
        type=str,
        required=True,
        help="KG directory to merge the MESH to.",
    )

    parser.add_argument(
        "--use_pkl",
        action="store_true",
        help="Set if using pickled data.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    # extract content
    df_content = pd.read_csv(CONTENT_FILEPATH, low_memory=False, keep_default_na=False)
    print(f"Original Content.csv file shape: {df_content.shape[0]}")
    df_compound = df_content[df_content["source_type"] == "Compound"]

    citation_type = list(set(df_compound["citation_type"].tolist()))
    print("citation_types: ", citation_type)

    df_compound = df_compound[df_compound["export"] == 1]
    df_compound = df_compound[df_compound["citation_type"] == "EXPERIMENTAL"]
    print(f"Compounds shape: {df_content.shape[0]}")

    # extract pairs
    pairs = list(zip(df_compound["food_id"].tolist(), df_compound["source_id"].tolist()))
    pairs = list(set(pairs))
    print(len(pairs))

    sys.exit()


    # food and compound mapping
    df_food = pd.read_csv(FOOD_FILEPATH, keep_default_na=False)
    df_food = df_food[df_food["ncbi_taxonomy_id"] != ""]
    food_id_ncbi_map = dict(zip(df_food["id"].tolist(), df_food["ncbi_taxonomy_id"]))

    df_compound = pd.read_csv(COMPOUND_FILEPATH, keep_default_na=False)
    print(df_compound)




    sys.exit()

    df_compound_subset = df_compound[df_compound["moldb_inchikey"] != ""]
    compound_id_inchi_map = dict(zip(
        df_compound_subset["id"].tolist(), df_compound_subset["moldb_inchikey"]))

    df_compound_subset = df_compound[df_compound["description"] != ""]
    compound_id_cas_map = dict(zip(
        df_compound_subset["id"].tolist(), df_compound_subset["description"]))

    # merge foodbid


if __name__ == '__main__':
    main()
