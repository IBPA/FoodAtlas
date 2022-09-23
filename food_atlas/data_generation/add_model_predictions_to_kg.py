import argparse
import os
from pathlib import Path
import random
import sys

sys.path.append('..')

import pandas as pd  # noqa: E402

from common_utils.utils import read_tsv  # noqa: E402
from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402


PREDICTED_FILEPATH = "../../outputs/data_generation/predicted_{}.tsv"
THRESHOLD = 0.5
KG_OUTPUT_DIR = "../../outputs/kg"
KG_FILENAME = "kg.txt"
ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="What pre_annotation round are we generating.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Threshold cutoff (Default: {THRESHOLD})",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    # read predicted
    predicted_filepath = PREDICTED_FILEPATH.format(args.round)
    df_pred = read_tsv(predicted_filepath)
    print(f"Predicted shape: {df_pred.shape}")
    # df_pred = df_pred.head(500)

    # add predictions
    output_dir = os.path.join(KG_OUTPUT_DIR, str(args.round))
    fa_kg = KnowledgeGraph(
        kg_filepath=os.path.join(output_dir, KG_FILENAME),
        entities_filepath=os.path.join(output_dir, ENTITIES_FILENAME),
        relations_filepath=os.path.join(output_dir, RELATIONS_FILENAME),
    )

    print(fa_kg.num_entities())

    from time import time
    start = time()

    fa_kg.add_ph_pairs(df_pred)

    end = time()
    print(end-start)

    # fa_kg.save()

    print(fa_kg.num_entities())
    print(fa_kg.avail_entity_id)


if __name__ == '__main__':
    main()
