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
OUTPUT_DIR = "../../outputs/analysis_codes/figure6"
NAMES = ['head', 'relation', 'tail']


def get_df():
    df_top_100 = pd.read_csv(
        os.path.join('../../outputs/hypotheses/validated_top_100.tsv'),
        sep='\t',
    )

    df_bottom_100 = pd.read_csv(
        os.path.join('../../outputs/hypotheses/validated_bottom_100.tsv'),
        sep='\t',
    )

    df_random_100 = pd.read_csv(
        os.path.join('../../outputs/hypotheses/validated_random_100.tsv'),
        sep='\t',
    )

    df = pd.concat([df_top_100, df_bottom_100, df_random_100])
    df.sort_values('prob_mean', inplace=True, ascending=False)

    return df


def plot_calibration_plot(df):
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lineplot_x = []
    lineplot_y = []

    barplot_num_pos = []
    barplot_num_neg = []
    for idx_low in range(len(bins)-1):
        low_prob = bins[idx_low]
        high_prob = bins[idx_low+1]

        if high_prob != 1.0:
            df_bin = df[df['prob_mean'].apply(lambda x: x >= low_prob and x < high_prob)]
        else:
            df_bin = df[df['prob_mean'].apply(lambda x: x >= low_prob)]

        validation_labels = df_bin['Validation'].tolist()
        validation_labels = [x for x in validation_labels if x != 'NER error']
        num_pos = validation_labels.count('Yes')
        fraction_pos = num_pos / len(validation_labels)
        # print(f'{low_prob}, {high_prob}: {num_pos}/{len(validation_labels)} ({fraction_pos})')

        lineplot_x.append(float(f'{(low_prob + high_prob)/2:.1f}'))
        lineplot_y.append(fraction_pos)
        barplot_num_pos.append(num_pos)
        barplot_num_neg.append(len(validation_labels) - num_pos)

    df_bar = pd.DataFrame({'pos': barplot_num_pos, 'neg': barplot_num_neg})
    df_bar['x'] = lineplot_x

    # r2
    print(lineplot_x)
    print(lineplot_y)
    r2 = r2_score(lineplot_x, lineplot_y)
    print(f'R2: {r2}')

    # line
    plt.plot(lineplot_x, lineplot_y)
    plt.plot(lineplot_x, lineplot_x)
    plt.text(0.2, 0.8, str(r2))
    plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_plot.png'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_plot.svg'))

    plot = df_bar.plot(x='x', kind='bar', stacked=False)
    plot.get_figure().savefig(os.path.join(OUTPUT_DIR, 'calibration_plot_bar.png'))
    plot.get_figure().savefig(os.path.join(OUTPUT_DIR, 'calibration_plot_bar.svg'))


def generate_gephi(df, fa_kg):
    df_entities = fa_kg.get_all_entities()
    df_relations = fa_kg.get_all_relations()
    rel_dict = dict(zip(df_relations['foodatlas_id'], df_relations['name']))

    df_pos = df[df['Validation'] == 'Yes']
    df_pos.drop_duplicates(['head', 'relation', 'tail'], inplace=True)
    print(df_pos.shape)
    foods = set(df_pos['head'].tolist())
    chemicals = set(df_pos['tail'].tolist())
    print(len(foods))
    print(len(chemicals))

    sys.exit()

    edge_data = []
    entities = []
    for _, row in df_pos.iterrows():
        entities.extend([row['head'], row['tail']])
        source = fa_kg.get_entity_by_id(row['head'])['name']
        target = fa_kg.get_entity_by_id(row['tail'])['name']
        label = rel_dict[row['relation']]
        weight = 0.01 if row['prob_mean'] < 0.01 else row['prob_mean']
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


def plot_hypotheses_bins(fa_kg):
    df = pd.read_csv(
        '../../outputs/kgc/pykeen/annotations_predictions-' +
        'above80_extdb_mesh_ncbi/hpo/RotatE/hypotheses_20230323_010058.tsv',
        sep='\t')
    print(df)

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    x = []
    barplot_num = []
    for idx_low in range(len(bins)-1):
        low_prob = bins[idx_low]
        high_prob = bins[idx_low+1]
        print(f'{low_prob}, {high_prob}')

        if high_prob != 1.0:
            df_bin = df[df['prob_mean'].apply(lambda x: x >= low_prob and x < high_prob)]
        else:
            df_bin = df[df['prob_mean'].apply(lambda x: x >= low_prob)]

        x.append(f'{low_prob}, {high_prob}')
        barplot_num.append(df_bin.shape[0])

    df = pd.DataFrame({'x': x, 'val': barplot_num})
    print(df)
    print(np.sum(df['val'].tolist()))

    plot = df.plot.bar(x='x', y='val', rot=90, logy=True)
    plot.get_figure().savefig(os.path.join(OUTPUT_DIR, 'num_hypotheses.png'))
    plot.get_figure().savefig(os.path.join(OUTPUT_DIR, 'num_hypotheses.svg'))


def find_interesting_food_chem(fa_kg):
    df = pd.read_csv(
        '../../outputs/kgc/pykeen/annotations_predictions-' +
        'above80_extdb_mesh_ncbi/hpo/RotatE/hypotheses_20230323_010058.tsv',
        sep='\t')
    print(df)

    df_foods = df.groupby('head')['prob_mean'].apply(list).reset_index(name='probs')
    df_foods['mean'] = df_foods['probs'].apply(lambda x: np.mean(x))
    df_foods.sort_values('mean', ascending=False, inplace=True)
    df_foods['name'] = df_foods['head'].apply(lambda x: fa_kg.get_entity_by_id(x)['name'])
    df_foods = df_foods.head(5)
    print(df_foods)

    df_chems = df.groupby('tail')['prob_mean'].apply(list).reset_index(name='probs')
    df_chems['mean'] = df_chems['probs'].apply(lambda x: np.mean(x))
    df_chems.sort_values('mean', ascending=False, inplace=True)
    df_chems['name'] = df_chems['tail'].apply(lambda x: fa_kg.get_entity_by_id(x)['name'])
    df_chems = df_chems.head(5)
    print(df_chems)

    fig = go.Figure()
    for _, row in df_foods.iterrows():
        fig.add_trace(go.Violin(
            x=row['probs'],
            box_visible=True,
            meanline_visible=True,
            name=row['name']),
        )

    for _, row in df_chems.iterrows():
        fig.add_trace(go.Violin(
            x=row['probs'],
            box_visible=True,
            meanline_visible=True,
            name=row['name']),
        )

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        font_family="Arial",
        height=700, width=700,
    )
    fig.write_image(os.path.join(OUTPUT_DIR, "food_chem_boxplot.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "food_chem_boxplot.svg"))


def known_unknown_entities(fa_kg):
    df = pd.read_csv(
        '../../outputs/kgc/pykeen/annotations_predictions-' +
        'above80_extdb_mesh_ncbi/hpo/RotatE/hypotheses_20230323_010058.tsv',
        sep='\t')
    print(df)

    foods = list(set(df['head'].tolist()))
    chemicals = list(set(df['tail'].tolist()))

    num_foods = len(foods)
    num_chemicals = len(chemicals)
    print(num_foods)
    print(num_chemicals)

    # foods
    df_foods = df.groupby('head')['tail'].apply(len).reset_index(name='num_unknown_chems')
    df_foods['num_known_chems'] = df_foods['num_unknown_chems'].apply(lambda x: num_chemicals-x)
    df_foods.sort_values('num_known_chems', ascending=True, inplace=True)
    df_foods['name'] = df_foods['head'].apply(lambda x: fa_kg.get_entity_by_id(x)['name'])
    print(df_foods)

    num_known_chems = Counter(df_foods['num_known_chems'].to_list())
    print(f'Number of foods with only 1 known chem: {num_known_chems[1]}')
    print(f'Number of foods with only 2 known chem: {num_known_chems[2]}')

    df_foods = df_foods.tail()
    print(df_foods)

    #
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_foods['name'],
        y=df_foods['num_unknown_chems'],
        name='unknown',
    ))
    fig.add_trace(go.Bar(
        x=df_foods['name'],
        y=df_foods['num_known_chems'],
        name='known',
    ))
    fig.update_layout(barmode='relative')

    fig.write_image(os.path.join(OUTPUT_DIR, "well_known_food.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "well_known_food.svg"))

    # chems
    df_chems = df.groupby('tail')['head'].apply(len).reset_index(name='num_unknown_foods')
    df_chems['num_known_foods'] = df_chems['num_unknown_foods'].apply(lambda x: num_foods-x)
    df_chems.sort_values('num_known_foods', ascending=True, inplace=True)
    df_chems['name'] = df_chems['tail'].apply(lambda x: fa_kg.get_entity_by_id(x)['name'])
    print(df_chems)

    num_known_foods = Counter(df_chems['num_known_foods'].to_list())
    print(f'Number of chems with only 1 known food: {num_known_foods[1]}')
    print(f'Number of chems with only 2 known food: {num_known_foods[2]}')

    df_chems = df_chems.tail()
    print(df_chems)

    #
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_chems['name'],
        y=df_chems['num_unknown_foods'],
        name='unknown',
    ))
    fig.add_trace(go.Bar(
        x=df_chems['name'],
        y=df_chems['num_known_foods'],
        name='known',
    ))
    fig.update_layout(barmode='relative')

    fig.write_image(os.path.join(OUTPUT_DIR, "well_known_chem.png"))
    fig.write_image(os.path.join(OUTPUT_DIR, "well_known_chem.svg"))


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fa_kg = KnowledgeGraph(kg_dir=KG_DIR)

    df = get_df()
    # plot_calibration_plot(df)
    generate_gephi(df, fa_kg)
    # plot_hypotheses_bins(fa_kg)
    # find_interesting_food_chem(fa_kg)
    # known_unknown_entities(fa_kg)


if __name__ == '__main__':
    main()
