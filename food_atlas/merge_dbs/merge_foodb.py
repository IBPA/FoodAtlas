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

    df_content = pd.read_csv(CONTENT_FILEPATH, low_memory=False, keep_default_na=False)
    df_compound = df_content[df_content["source_type"] == "Compound"]

    citation_type = list(set(df_compound["citation_type"].tolist()))
    print("citation_type: ", citation_type)

    df_compound = df_compound[df_compound["citation_type"] == "DATABASE"]

    citation = list(set(df_compound["citation"].tolist()))
    print("citation: ", citation)

    #
    # df_compound = df_compound[df_compound["citation_type"].apply(lambda x: x in ['EXPERIMENTAL'])]

    df_compound.to_csv('./temp.csv', sep='\t', index=False)


if __name__ == '__main__':
    main()
