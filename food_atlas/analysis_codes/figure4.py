import os
from pathlib import Path
import sys

sys.path.append('../data_processing/')

from tqdm import tqdm  # noqa: E402
import pandas as pd  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

OUTPUT_DIR = "../../outputs/analysis_codes/figure4"
KG_DIR = "../../outputs/backend_data/v0.1"

MESH_DATA_DIR = "../../data/MESH"
DESC_FILEPATH = "../../data/MESH/desc2022.xml"
SUPP_FILEPATH = "../../data/MESH/supp2022.xml"


def generate_sub_gephi(
        fa_kg,
        targt_ncbi_taxonomy,
):
    #
    df_kg = fa_kg.get_kg()
    df_entities = fa_kg.get_all_entities()
    df_relations = fa_kg.get_all_relations()
    rel_dict = dict(zip(df_relations['foodatlas_id'], df_relations['name']))

    df_subs = fa_kg.get_entity_by_other_db_id({'NCBI_taxonomy': [targt_ncbi_taxonomy]})
    sub_foodatlas_ids = set(df_subs['foodatlas_id'].tolist())

    # EDGE for (sub, contains, chem) (sub, hasPart, sub parts)
    entities = []
    edge_data = []
    for _, row in tqdm(df_kg.iterrows(), total=df_kg.shape[0]):
        if row['head'] not in sub_foodatlas_ids:
            continue
        source = fa_kg.get_entity_by_id(row['head'])['name']
        target = fa_kg.get_entity_by_id(row['tail'])['name']
        label = rel_dict[row['relation']]
        edge_data.append([source, target, label, label])
        entities.extend([row['head'], row['tail']])

    df_edge = pd.DataFrame(edge_data, columns=['Source', 'Target', 'Label', 'Label_copy'])
    df_edge['Weight'] = 1
    df_edge['Type'] = 'Directed'
    print(df_edge)

    if targt_ncbi_taxonomy == '3641':
        food_name = 'cocoa'
    elif targt_ncbi_taxonomy == '4682':
        food_name = 'garlic'
    else:
        raise ValueError()

    df_edge.to_csv(
        os.path.join(OUTPUT_DIR, f'{food_name}_edges.csv'),
        sep='\t', index=False,
    )

    nodes_data = []
    for _, row in tqdm(df_entities.iterrows(), total=df_entities.shape[0]):
        if row['foodatlas_id'] not in entities:
            continue
        node_id = row['name']
        node_type = row['type'].split(':')[0]
        nodes_data.append([node_id, node_id, node_type])

    df_nodes = pd.DataFrame(nodes_data, columns=['Id', 'Label', 'Type'])
    print(df_nodes)

    df_nodes.to_csv(
        os.path.join(OUTPUT_DIR, f'{food_name}_nodes.csv'),
        sep='\t', index=False,
    )


def main():
    fa_kg = KnowledgeGraph(kg_dir=KG_DIR)
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    # generate_sub_gephi(fa_kg, '4682')
    # generate_sub_gephi(fa_kg, '3641')


if __name__ == '__main__':
    main()
