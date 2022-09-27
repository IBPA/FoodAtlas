import argparse
import sys

sys.path.append('..')

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402


KG_FILEPATH = "../../outputs/kg/{}/kg.txt"
EVIDENCE_FILEPATH = "../../outputs/kg/{}/evidence.txt"
ENTITIES_FILEPATH = "../../outputs/kg/{}/entities.txt"
RELATIONS_FILEPATH = "../../outputs/kg/{}/relations.txt"


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
    fa_kg = KnowledgeGraph(
        kg_filepath=KG_FILEPATH.format(args.round),
        evidence_filepath=EVIDENCE_FILEPATH.format(args.round),
        entities_filepath=ENTITIES_FILEPATH.format(args.round),
        relations_filepath=RELATIONS_FILEPATH.format(args.round),
    )

    df_kg = fa_kg.get_kg()
    G = nx.DiGraph()
    for idx, row in tqdm(df_kg.iterrows(), total=df_kg.shape[0]):
        G.add_edge(row["head"], row["tail"])

    # create entity lookup dict
    df_entities = fa_kg.get_all_entities()
    dictionary = dict(zip(df_entities["foodatlas_id"], df_entities["name"]))

    return G, dictionary


def translate_path(path: list, dictionary: dict):
    return [dictionary[x] for x in path]


def main():
    args = parse_argument()

    G, dictionary = generate_kg(args)

    path = nx.shortest_path(G, source=3050, target=656)
    translated_path = translate_path(path, dictionary)
    print(path)
    print(translated_path)


if __name__ == '__main__':
    main()
