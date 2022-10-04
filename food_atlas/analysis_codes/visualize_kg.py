import argparse
from collections import Counter
import sys
import datetime

sys.path.append('..')

import graphistry  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd   # noqa: E402
from pandarallel import pandarallel   # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph, _ENTITY_TYPES  # noqa: E402

KG_FILENAME = "../../outputs/kg/{}/kg.txt"
EVIDENCE_FILENAME = "../../outputs/kg/{}/evidence.txt"
ENTITIES_FILENAME = "../../outputs/kg/{}/entities.txt"
RELATIONS_FILENAME = "../../outputs/kg/{}/relations.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="What pre_annotation round are we generating.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    pandarallel.initialize(progress_bar=True)

    graphistry.register(
        api=3,
        protocol="https",
        server="hub.graphistry.com",
        username="jasonyoun",
        password="foodatlas!@#$",
    )

    # load and process kg
    fa_kg = KnowledgeGraph(
        kg_filepath=KG_FILENAME.format(args.round),
        evidence_filepath=EVIDENCE_FILENAME.format(args.round),
        entities_filepath=ENTITIES_FILENAME.format(args.round),
        relations_filepath=RELATIONS_FILENAME.format(args.round),
    )

    df_kg = fa_kg.get_kg()
    df_evidence = fa_kg.get_evidence()

    def _get_sources(row):
        triple_str = f"({row['head']},{row['relation']},{row['tail']})"
        evidences = df_evidence[df_evidence["triple"] == triple_str]
        sources = evidences["source"].tolist()
        return dict(Counter(sources))
    df_kg["sources"] = df_kg.parallel_apply(lambda row: _get_sources(row), axis=1)

    def _get_prob(row):
        check_only_prediction = [
            True if x.startswith("prediction:") else False
            for x in row["sources"].keys()
        ]
        if False in check_only_prediction:
            return 1.0
        else:
            triple_str = f"({row['head']},{row['relation']},{row['tail']})"
            evidences = df_evidence[df_evidence["triple"] == triple_str]
            probs = evidences["prob"].tolist()
            return np.mean(probs)
    df_kg["prob"] = df_kg.parallel_apply(lambda row: _get_prob(row), axis=1)
    df_kg["sources"] = df_kg["sources"].astype(str)

    df_kg["head"] = df_kg["head"].map(lambda x: fa_kg.get_entity_by_id(x)["name"])
    df_kg["relation"] = df_kg["relation"].map(lambda x: fa_kg.get_relation_by_id(x)["name"])
    df_kg["tail"] = df_kg["tail"].map(lambda x: fa_kg.get_entity_by_id(x)["name"])

    df_entities = fa_kg.get_all_entities()
    df_entities["specific_type"] = df_entities["type"]
    df_entities["type"] = df_entities["type"].map(lambda x: x.split(":")[0])
    df_entities["synonyms"] = df_entities["synonyms"].astype(str)
    df_entities["other_db_ids"] = df_entities["other_db_ids"].astype(str)

    # visualize
    categorical_mapping = {}
    for x in _ENTITY_TYPES:
        if x.startswith("chemical"):
            categorical_mapping[x] = "rgb(66,135,245)"
        elif x.startswith("organism_with_part"):
            categorical_mapping[x] = "rgb(252,186,3)"
        elif x.startswith("organism"):
            categorical_mapping[x] = "rgb(50,168,82)"
        else:
            raise ValueError()

    g = (
        graphistry
        .edges(df_kg, "head", "tail")
        .nodes(df_entities, "name")
        .bind(
            edge_title="relation",
            edge_weight="prob",
        )
        .encode_point_color(
            "type",
            categorical_mapping=categorical_mapping,
        )
    )

    g.plot()


if __name__ == '__main__':
    main()
