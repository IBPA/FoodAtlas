import os
from pathlib import Path
import sys

sys.path.append('../data_processing/')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

OUTPUT_DIR = "../../outputs/analysis_codes/figure1"
KG_DIR = "../../outputs/backend_data/v0.1"


NODE_STR = """
    node [
        id {}
        label "{}"
        type "{}"
    ]"""

EDGE_STR = """
    edge [
        source {}
        target {}
        label "{}"
    ]"""


def generate_full_gml(fa_kg):
    df_kg = fa_kg.get_kg()
    df_entities = fa_kg.get_all_entities()
    df_relations = fa_kg.get_all_relations()
    rel_dict = dict(zip(df_relations['foodatlas_id'], df_relations['name']))

    content = "graph [\n    directed 1"

    for _, row in tqdm(df_entities.iterrows(), total=df_entities.shape[0]):
        node_id = int(row['foodatlas_id'][1:])
        # node_label = row['foodatlas_id'][1:] + ':' + row['name'].replace('"', "'")
        node_label = row['foodatlas_id'][1:]
        node_type = row['type'].split(':')[0]
        node = NODE_STR.format(node_id, node_label, node_type)
        content += node

    for _, row in tqdm(df_kg.iterrows(), total=df_kg.shape[0]):
        source = int(row['head'][1:])
        target = int(row['tail'][1:])
        label = rel_dict[row['relation']]
        edge = EDGE_STR.format(source, target, label)
        content += edge

    content += "\n]"

    with open(os.path.join(OUTPUT_DIR, 'kg.gml'), 'w') as _f:
        _f.write(content)


def generate_full_gephi(fa_kg):
    df_kg = fa_kg.get_kg()
    df_entities = fa_kg.get_all_entities()
    df_relations = fa_kg.get_all_relations()
    rel_dict = dict(zip(df_relations['foodatlas_id'], df_relations['name']))

    edge_data = []
    for _, row in tqdm(df_kg.iterrows(), total=df_kg.shape[0]):
        # source = fa_kg.get_entity_by_id(row['head'])['name'] + ':' + row['head']
        # target = fa_kg.get_entity_by_id(row['tail'])['name'] + ':' + row['tail']
        source = row['head']
        target = row['tail']
        label = rel_dict[row['relation']]
        edge_data.append([source, target, label, label])

    df_edge = pd.DataFrame(edge_data, columns=['Source', 'Target', 'Label', 'Label_copy'])
    df_edge['Weight'] = 1
    df_edge['Type'] = 'Directed'
    print(df_edge)

    df_edge.to_csv(
        os.path.join(OUTPUT_DIR, 'full_edges.csv'),
        sep='\t', index=False,
    )

    nodes_data = []
    for _, row in tqdm(df_entities.iterrows(), total=df_entities.shape[0]):
        node_id = row['foodatlas_id']
        node_type = row['type'].split(':')[0]
        nodes_data.append([node_id, node_id, node_type])

    df_nodes = pd.DataFrame(nodes_data, columns=['Id', 'Label', 'Type'])
    print(df_nodes)

    df_nodes.to_csv(
        os.path.join(OUTPUT_DIR, 'full_nodes.csv'),
        sep='\t', index=False,
    )


def generate_partial_gephi(fa_kg):
    df_kg = fa_kg.get_kg()
    df_entities = fa_kg.get_all_entities()
    df_relations = fa_kg.get_all_relations()
    rel_dict = dict(zip(df_relations['foodatlas_id'], df_relations['name']))

    df_kg = df_kg.sample(frac=0.1)
    entities = set(df_kg['head'].tolist() + df_kg['tail'].tolist())

    edge_data = []
    for _, row in tqdm(df_kg.iterrows(), total=df_kg.shape[0]):
        source = fa_kg.get_entity_by_id(row['head'])['name'] + ':' + row['head']
        target = fa_kg.get_entity_by_id(row['tail'])['name'] + ':' + row['tail']
        # source = row['head']
        # target = row['tail']
        label = rel_dict[row['relation']]
        edge_data.append([source, target, label, label])

    df_edge = pd.DataFrame(edge_data, columns=['Source', 'Target', 'Label', 'Label_copy'])
    df_edge['Weight'] = 1
    df_edge['Type'] = 'Directed'
    print(df_edge)

    df_edge.to_csv(
        os.path.join(OUTPUT_DIR, 'partial_edges.csv'),
        sep='\t', index=False,
    )

    nodes_data = []
    for _, row in tqdm(df_entities.iterrows(), total=df_entities.shape[0]):
        # node_id = row['foodatlas_id']
        node_id = fa_kg.get_entity_by_id(row['foodatlas_id'])['name'] + ':' + row['foodatlas_id']

        if row['foodatlas_id'] not in entities:
            continue

        node_type = row['type'].split(':')[0]
        nodes_data.append([node_id, node_id, node_type])

    df_nodes = pd.DataFrame(nodes_data, columns=['Id', 'Label', 'Type'])
    print(df_nodes)

    df_nodes.to_csv(
        os.path.join(OUTPUT_DIR, 'partial_nodes.csv'),
        sep='\t', index=False,
    )


def main():
    fa_kg = KnowledgeGraph(kg_dir=KG_DIR)
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    # generate_full_gml(fa_kg)
    # generate_full_gephi(fa_kg)
    generate_partial_gephi(fa_kg)


if __name__ == '__main__':
    main()
