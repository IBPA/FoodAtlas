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

    # degree centrality
    # The number of edges it has (either in or out).
    # Higher = more central.
    degree_centrality = nx.degree_centrality(G)
    df_degree_centrality = pd.DataFrame({
        "foodatlas_id": degree_centrality.keys(),
        "degree_centrality": degree_centrality.values(),
    })
    df_degree_centrality.sort_values("degree_centrality", ascending=False, inplace=True)
    print(df_degree_centrality)

    # betweenness centrality
    # The number of shortest paths between all pair
    # of nodes that pass through the node of interest.
    # Higher = More control over the network
    # since more information passes through this node.
    betweenness_centrality = nx.betweenness_centrality(G)
    df_betweenness_centrality = pd.DataFrame({
        "foodatlas_id": betweenness_centrality.keys(),
        "betweenness_centrality": betweenness_centrality.values(),
    })
    df_betweenness_centrality.sort_values("betweenness_centrality", ascending=False, inplace=True)
    print(df_betweenness_centrality)

    # closeness centrality
    # Average length of the shortest path between the
    # node and all other nodes in the graph.
    # Higher = close to all other nodes
    closeness_centrality = nx.closeness_centrality(G)
    df_closeness_centrality = pd.DataFrame({
        "foodatlas_id": closeness_centrality.keys(),
        "closeness_centrality": closeness_centrality.values(),
    })
    df_closeness_centrality.sort_values("closeness_centrality", ascending=False, inplace=True)
    print(df_closeness_centrality)


if __name__ == '__main__':
    main()
