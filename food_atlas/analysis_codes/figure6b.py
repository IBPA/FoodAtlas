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
import plotly.express as px

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402


KG_DIR = "../../outputs/backend_data/v0.1"
OUTPUT_DIR = "../../outputs/analysis_codes/figure6"
NAMES = ['head', 'relation', 'tail']


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fa_kg = KnowledgeGraph(kg_dir=KG_DIR)

    df_kg = fa_kg.get_kg()
    df_kg_contains = df_kg[df_kg['relation'] == 'r0']
    print(df_kg_contains)

    df_entities = fa_kg.get_all_entities()
    df_entities['type'] = df_entities['type'].apply(
        lambda x: 'organism' if x.startswith('organism') else 'others')
    id_name_lookup = dict(zip(df_entities['foodatlas_id'], df_entities['name']))

    df_chemical_groups = pd.read_csv('../../outputs/backend_data/v0.1/chemicals_group.txt', sep='\t')
    id_group_lookup = dict(zip(df_chemical_groups['foodatlas_id'], df_chemical_groups['group']))

    df_kg_contains['head'] = df_kg_contains['head'].apply(
        lambda x: id_name_lookup[x].split(' - ')[0])
    df_kg_contains.drop_duplicates(inplace=True)
    df_kg_contains['tail'] = df_kg_contains['tail'].apply(
        lambda x: id_group_lookup[x])
    print(df_kg_contains)

    df_grouped = df_kg_contains.groupby(['head', 'tail']).size().reset_index(name='count')
    print(df_grouped)

    # all food vs chem figure
    fig = px.bar(df_grouped, x='head', y='count', color='tail')
    fig.update_xaxes(
        showticklabels=False,
    )
    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        font_family="Arial",
        height=700, width=1500,
        xaxis={'categoryorder': 'total descending'},
    )
    fig.write_image(os.path.join(OUTPUT_DIR, "food_chemicals.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "food_chemicals.svg"))

    # top 5
    df_top_5 = df_grouped.groupby(['head']).sum().reset_index()
    df_top_5.sort_values('count', ascending=False, inplace=True)
    top_5_foods = df_top_5.head(5)['head'].tolist()
    print(top_5_foods)

    df_grouped = df_grouped[df_grouped['head'].apply(lambda x: x in top_5_foods)]
    print(df_grouped)

    # top 5 food vs chem figure
    fig = px.bar(df_grouped, x='head', y='count', color='tail')
    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        font_family="Arial",
        height=700, width=1000,
        xaxis={'categoryorder': 'total descending'},
    )
    fig.write_image(os.path.join(OUTPUT_DIR, "food_chemicals_top5.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "food_chemicals_top5.svg"))


if __name__ == '__main__':
    main()
