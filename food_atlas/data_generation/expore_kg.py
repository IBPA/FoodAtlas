import argparse
from collections import Counter
from glob import glob
from pathlib import Path
import random
import sys

sys.path.append('..')

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.foodatlas_types import FoodAtlasEntity, CandidateEntity # noqa: E402
from common_utils.utils import read_kg, read_tsv  # noqa: E402


KG_FILEPATH = "../../outputs/kg/{}/kg.txt"
ENTITIES_FILEPATH = "../../outputs/kg/{}/entities.txt"
RELATIONS_FILEPATH = "../../outputs/kg/{}/relations.txt"
PREDICTED_FILEPATH = "../../outputs/data_generation/predicted_{}.tsv"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="What round of KG are we exploring.",
    )

    args = parser.parse_args()
    return args


def generate_kg(args):
    kg_filepath = KG_FILEPATH.format(args.round)
    entities_filepath = ENTITIES_FILEPATH.format(args.round)
    relations_filepath = RELATIONS_FILEPATH.format(args.round)
    predicted_filepath = PREDICTED_FILEPATH.format(args.round)

    # this is the current round kg
    df_kg = read_kg(kg_filepath)
    fa_ent = FoodAtlasEntity(entities_filepath)
    print(f"Number of entities in KG: {fa_ent.num_entities()}")

    G = nx.DiGraph()
    for idx, row in df_kg.iterrows():
        G.add_edge(row["head"], row["tail"])

    # get all candidate entities and add them
    df_predicted = read_tsv(predicted_filepath)

    # df_predicted = df_predicted.head(1000)

    # for idx, row in df_predicted.iterrows():


    # nx.draw_networkx(G, pos, **options)
    # ax = plt.gca()
    # ax.margins(0.20)
    # plt.axis("off")
    # plt.show()


def main():
    args = parse_argument()

    generate_kg(args)


if __name__ == '__main__':
    main()
