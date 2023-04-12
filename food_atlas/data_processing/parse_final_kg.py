import argparse
import math
import os
import sys
import warnings

import pandas as pd
from tqdm import tqdm

from common_utils.knowledge_graph import KnowledgeGraph
from common_utils.utils import load_pkl
from common_utils.chemical_db_ids import (
    _get_name_from_json, _get_summary_description_from_json,
    read_mesh_data, get_mesh_name_using_mesh_id
)
from merge_ncbi_taxonomy import read_dmp_files


MESH_DATA_DIR = "../../data/MESH"
FOODB_FOOD_FILEPATH = "../../data/FooDB/foodb_2020_04_07_csv/Food.csv"
ORGANISMS_GROUP_FILENAME = "organisms_group.txt"
CHEMICALS_GROUP_FILENAME = "chemicals_group.txt"

NODES_FILEPATH = "../../data/NCBI_Taxonomy/nodes.dmp"
NAMES_FILEPATH = "../../data/NCBI_Taxonomy/names.dmp"
CID_JSON_LOOKUP_PKL_FILEPATH = "../../data/FoodAtlas/cid_json_lookup.pkl"
PARTS_FILEPATH = "../../data/FoodAtlas/food_parts.txt"
MESH_DATA_DIR = "../../data/MESH"
DESC_FILEPATH = "../../data/MESH/desc2022.xml"
SUPP_FILEPATH = "../../data/MESH/supp2022.xml"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        required=True,
        help="Final KG dir",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output dir.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    # read KG and save all files to the new folder
    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir)
    df_entities = fa_kg.get_all_entities()

    # 

    # update entity names
    print("Reading names.dmp...")
    df_names = read_dmp_files(NAMES_FILEPATH, filetype="names")
    df_names = df_names[df_names['name_class'] == 'scientific name']
    ncbi_taxid_name_lookup = dict(zip(df_names['tax_id'], df_names['name_txt']))

    print('Loading cid_json_lookup...')
    cid_json_lookup = load_pkl(CID_JSON_LOOKUP_PKL_FILEPATH)
    print('Loading mesh_data_dict...')
    mesh_data_dict = read_mesh_data(
        mesh_data_dir=MESH_DATA_DIR, desc_filepath=DESC_FILEPATH, supp_filepath=SUPP_FILEPATH)

    print('Loading food parts...')
    df_food_parts = pd.read_csv(PARTS_FILEPATH, sep='\t')
    food_parts_lookup = dict(zip(df_food_parts['foodatlas_part_id'], df_food_parts['food_part']))

    print("Updating entities...")
    unknown_ncbi_taxids = []
    for _, row in tqdm(df_entities.iterrows(), total=df_entities.shape[0]):
        other_db_ids = row['other_db_ids']
        if row['type'] == 'chemical':
            if 'PubChem' in other_db_ids:
                json = cid_json_lookup[other_db_ids['PubChem'][0]]
                new_name = _get_name_from_json(json)
            elif 'MESH' in other_db_ids:
                mesh_id = other_db_ids['MESH'][0]
                new_name = get_mesh_name_using_mesh_id(mesh_id, mesh_data_dict)
            else:
                raise ValueError()
        elif row['type'] == 'organism_with_part' or row['type'].startswith('organism_with_part:'):
            query_id = row["other_db_ids"]["NCBI_taxonomy"]
            assert len(query_id) == 1
            query_id = query_id[0]

            foodatlas_part_id = other_db_ids['foodatlas_part_id']
            part_name = food_parts_lookup[foodatlas_part_id]

            if query_id in ncbi_taxid_name_lookup:
                new_name = f'{ncbi_taxid_name_lookup[query_id]} - {part_name}'
            else:
                unknown_ncbi_taxids.append(query_id)
        elif row['type'] == 'organism' or row['type'].startswith('organism:'):
            query_id = row["other_db_ids"]["NCBI_taxonomy"]
            assert len(query_id) == 1
            query_id = query_id[0]

            if query_id in ncbi_taxid_name_lookup:
                new_name = ncbi_taxid_name_lookup[query_id]
            else:
                unknown_ncbi_taxids.append(query_id)
        else:
            raise ValueError()

        if new_name != row['name']:
            fa_kg._overwrite_entity(
                foodatlas_id=row["foodatlas_id"],
                type_=row['type'],
                name=new_name,
            )

    print(f'Unknown NCBI taxonomy IDs: {set(unknown_ncbi_taxids)}')

    fa_kg.save(kg_dir=args.output_dir)


if __name__ == '__main__':
    main()
