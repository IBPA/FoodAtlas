from ast import literal_eval
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel

from ..data_processing.common_utils.knowledge_graph import KnowledgeGraph

pandarallel.initialize(progress_bar=True)


if __name__ == '__main__':
    kg = KnowledgeGraph(kg_dir="outputs/backend_data/v0.1")
    evidence = kg.get_evidence()
    evidence = evidence.query(
        "pmid != '' & relation == 'r0' & "
        "source in ['FoodAtlas:prediction:entailment', 'FoodAtlas:annotation']"
    )
    evidence.loc[evidence['quality'] == 'high', 'prob_mean'] = 1.0

    # Group by PMID and unique hypotheses.
    def get_n_triples(group):
        # group = group.sort_values(
        #     ['triple', 'quality', 'prob_mean'],
        #     ascending=[True, True, False],
        # )
        # print(group)

        return pd.Series({
            'triple': group['triple'].iloc[0],
            'n_triples': group['prob_mean'].astype(float).max(),
        })

    n_triplets = evidence.groupby(
        'triple', as_index=False
    ).parallel_apply(get_n_triples)
    n_triplets.to_csv("outputs/benchmark/n_triplets.csv", index=False)

    print(f"# triples from literature: {len(evidence)}")
    print(f"# unique triples from literature: {evidence['triple'].nunique()}")
    print(f"# unique triples weighted: {n_triplets['n_triples'].sum()}")
    print(f"# unique PMIDs: {evidence['pmid'].nunique()}")

    # Estimate number of triples in the world.
    kg = KnowledgeGraph(kg_dir="outputs/backend_data/v0.1")
    evidence = kg.get_evidence()
    evidence = evidence.query(
        "pmid != '' & relation == 'r0' & "
        "source in ['FoodAtlas:prediction:entailment', 'FoodAtlas:annotation']"
    )
    n_triplets = pd.read_csv("outputs/benchmark/n_triplets.csv")

    # 1. Unique triples per article.
    n_trip_mean = n_triplets['n_triples'].sum() / evidence['pmid'].nunique()

    # 2. Estimate based on # of sources related to food in AGRICOLA. We
    # searched for "food" in any field.
    #  - 887,127 articles
    #  - 89,940 books
    #  - 6,961 conference proceedings
    #  - 4,469 book chapters
    n_sources = 887127 + 89940 + 6961 + 4469
    n_triplets_gt_est = n_trip_mean * n_sources
    print(n_triplets_gt_est)  # 1304284.202649675
