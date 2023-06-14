from ast import literal_eval
from collections import Counter
from glob import glob
import os
from pathlib import Path
import sys

sys.path.append('../data_processing/')

import seaborn as sns  # noqa: E402
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402


KG_DIR = "../../outputs/backend_data/v0.1"
OUTPUT_DIR = "../../outputs/analysis_codes/web"
NAMES = ['head', 'relation', 'tail']


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fa_kg = KnowledgeGraph(kg_dir=KG_DIR)

    df_kg = fa_kg.get_kg()
    df_kg = df_kg.sample(frac=0.05)

    df_relations = fa_kg.get_all_relations()
    rel_dict = dict(zip(df_relations['foodatlas_id'], df_relations['name']))

    edge_data = []
    entities = []
    for _, row in df_kg.iterrows():
        entities.extend([row['head'], row['tail']])
        source = fa_kg.get_entity_by_id(row['head'])['name']
        target = fa_kg.get_entity_by_id(row['tail'])['name']
        label = rel_dict[row['relation']]
        weight = 1.0
        edge_data.append([source, target, label, label, weight])

    df_edge = pd.DataFrame(edge_data, columns=['Source', 'Target', 'Label', 'Label_copy', 'Weight'])
    df_edge['Type'] = 'Directed'

    df_edge.to_csv(
        os.path.join(OUTPUT_DIR, 'edges.csv'),
        sep='\t', index=False,
    )

    nodes_data = []
    for foodatlas_id in set(entities):
        node_id = fa_kg.get_entity_by_id(foodatlas_id)['name']
        node_label = fa_kg.get_entity_by_id(foodatlas_id)['name']
        node_type = fa_kg.get_entity_by_id(foodatlas_id)['type'].split(':')[0]
        nodes_data.append([node_id, node_label, node_type])

    df_nodes = pd.DataFrame(nodes_data, columns=['Id', 'Label', 'Type'])

    df_nodes.to_csv(
        os.path.join(OUTPUT_DIR, 'nodes.csv'),
        sep='\t', index=False,
    )


if __name__ == '__main__':
    main()
