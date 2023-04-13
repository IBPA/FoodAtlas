import argparse
import math
import os
from pathlib import Path
import sys
import warnings

import pandas as pd
from tqdm import tqdm

from common_utils.knowledge_graph import KnowledgeGraph
from common_utils.utils import load_pkl, save_pkl
from common_utils.chemical_db_ids import (
    _get_name_from_json, _get_summary_description_from_json,
    read_mesh_data, get_mesh_name_using_mesh_id,
    _get_inchi_from_json, _get_inchikey_from_json,
    _get_canonical_smiles_from_json,
    _get_synonyms_from_json,
)
from merge_ncbi_taxonomy import read_dmp_files


MESH_DATA_DIR = "../../data/MESH"
FOODB_FOOD_FILEPATH = "../../data/FooDB/foodb_2020_04_07_csv/Food.csv"
ORGANISMS_GROUP_FILENAME = "organisms_group.txt"
CHEMICALS_GROUP_FILENAME = "chemicals_group.txt"
CHEMICALS_IDENTIFIERS_FILENAME = "chemicals_identifiers.txt"
SYNONYMS_FILENAME = "synonyms.txt"

GROUPED_FOODS_FILEPATH = "../../data/NCBI_Taxonomy/grouped_foods.pkl"

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
    df_kg = fa_kg.get_kg()

    contains_relation = fa_kg.get_relation_by_name('contains')
    df_kg_contains = df_kg[df_kg['relation'] == contains_relation['foodatlas_id']]

    # update entity names
    print("Reading names.dmp...")
    df_names = read_dmp_files(NAMES_FILEPATH, filetype="names")

    print('Loading cid_json_lookup...')
    cid_json_lookup = load_pkl(CID_JSON_LOOKUP_PKL_FILEPATH)
    print('Loading mesh_data_dict...')
    mesh_data_dict = read_mesh_data(
        mesh_data_dir=MESH_DATA_DIR, desc_filepath=DESC_FILEPATH, supp_filepath=SUPP_FILEPATH)

    print('Loading food parts...')
    df_food_parts = pd.read_csv(PARTS_FILEPATH, sep='\t')
    food_parts_lookup = dict(zip(df_food_parts['foodatlas_part_id'], df_food_parts['food_part']))

    # synonyms
    print('Getting synonyms...')
    df_entities = fa_kg.get_all_entities()

    if Path(GROUPED_FOODS_FILEPATH).is_file():
        print(f'Found pickled dataframe for grouped foods: {GROUPED_FOODS_FILEPATH}')
        df_names_synonyms = load_pkl(GROUPED_FOODS_FILEPATH)
    else:
        print('Groupby tax_id... This takes a while...')
        df_names_synonyms = df_names.groupby('tax_id').agg(lambda x: list(x))
        df_names_synonyms = df_names_synonyms[['name_txt', 'unique_name']]
        df_names_synonyms['synonyms'] = df_names_synonyms.values.tolist()
        df_names_synonyms['synonyms'] = df_names_synonyms['synonyms'].apply(
            lambda x: [j for i in x for j in i if j != '']
        )
        df_names_synonyms.reset_index(inplace=True)
        save_pkl(df_names_synonyms, GROUPED_FOODS_FILEPATH)

    ncbi_synonym_lookup = dict(zip(
        df_names_synonyms['tax_id'], df_names_synonyms['synonyms']
    ))

    desc_mesh_id_entry_lookup = mesh_data_dict['desc_mesh_id_entry_lookup']
    supp_mesh_id_entry_lookup = mesh_data_dict['supp_mesh_id_entry_lookup']

    def get_synonyms(row):
        synonyms = []
        other_db_ids = row['other_db_ids']

        if row['type'].startswith('chemical'):
            if 'MESH' in other_db_ids:
                for mesh_id in other_db_ids['MESH']:
                    if mesh_id[0] == 'D' and mesh_id in desc_mesh_id_entry_lookup:
                        synonyms.extend(desc_mesh_id_entry_lookup[mesh_id])
                    elif mesh_id[0] == 'C' and mesh_id in supp_mesh_id_entry_lookup:
                        synonyms.extend(supp_mesh_id_entry_lookup[mesh_id])
                    else:
                        warnings.warn(f'Unsupported MeSH ID: {mesh_id}')

            if 'PubChem' in other_db_ids:
                for pubchem_id in other_db_ids['PubChem']:
                    if pubchem_id in cid_json_lookup:
                        synonyms.extend(_get_synonyms_from_json(cid_json_lookup[pubchem_id]))
        elif row['type'].startswith('organism'):
            if 'NCBI_taxonomy' in other_db_ids:
                for ncbi_taxonomy in other_db_ids['NCBI_taxonomy']:
                    if ncbi_taxonomy in ncbi_synonym_lookup:
                        raw_syn = ncbi_synonym_lookup[ncbi_taxonomy]

                        if row['type'].startswith('organism_with_part'):
                            foodatlas_part_id = other_db_ids['foodatlas_part_id']
                            part_name = food_parts_lookup[foodatlas_part_id]
                            raw_syn = [f'{x} - {part_name}' for x in raw_syn]

                        synonyms.extend(raw_syn)
            else:
                raise ValueError()
        else:
            raise ValueError()

        return sorted(set(synonyms))

    df_entities['synonyms'] = df_entities.apply(lambda row: get_synonyms(row), axis=1)
    df_entities = df_entities[['foodatlas_id', 'synonyms']]
    df_entities.to_csv(
        os.path.join(args.output_dir, SYNONYMS_FILENAME),
        sep='\t', index=False,
    )

    # chemical identifiers
    print('Getting chemical identifiers...')
    df_chemicals = fa_kg.get_entities_by_type(exact_type='chemical')
    df_chemicals['PubChem'] = df_chemicals['other_db_ids'].apply(
        lambda x: x['PubChem'] if 'PubChem' in x else None
    )

    df_chemicals = df_chemicals.explode('PubChem')
    df_chemicals['InChi'] = df_chemicals['PubChem'].apply(
        lambda x: _get_inchi_from_json(cid_json_lookup[x]) if x in cid_json_lookup else None
    )
    df_chemicals['InChiKey'] = df_chemicals['PubChem'].apply(
        lambda x: _get_inchikey_from_json(cid_json_lookup[x]) if x in cid_json_lookup else None
    )
    df_chemicals['SMILES'] = df_chemicals['PubChem'].apply(
        lambda x: _get_canonical_smiles_from_json(cid_json_lookup[x]) if x in cid_json_lookup
        else None
    )
    prev_size = df_chemicals.shape[0]
    df_chemicals = df_chemicals.explode('InChi')
    df_chemicals = df_chemicals.explode('InChiKey')
    df_chemicals = df_chemicals.explode('SMILES')
    assert df_chemicals.shape[0] == prev_size
    df_chemicals = df_chemicals[['foodatlas_id', 'PubChem', 'InChi', 'InChiKey', 'SMILES']]
    df_chemicals.to_csv(
        os.path.join(args.output_dir, CHEMICALS_IDENTIFIERS_FILENAME),
        sep='\t', index=False,
    )

    # chemical group and subgroup
    print('Getting chemical group and subgroup...')
    desc_mesh_id_tree_number_lookup = mesh_data_dict['desc_mesh_id_tree_number_lookup']
    desc_tree_number_name_lookup = mesh_data_dict['desc_tree_number_name_lookup']
    supp_mesh_id_heading_mesh_id_lookup = mesh_data_dict['supp_mesh_id_heading_mesh_id_lookup']

    df_chemicals = fa_kg.get_entities_by_type(exact_type='chemical')
    df_chemicals['MESH'] = df_chemicals['other_db_ids'].apply(
        lambda x: x['MESH'] if 'MESH' in x else None
    )
    df_chemicals = df_chemicals.explode('MESH')

    def _get_mesh_tree(mesh_id):
        if mesh_id is None:
            return None

        tree_numbers = []
        if mesh_id[0] == 'C':
            if mesh_id not in supp_mesh_id_heading_mesh_id_lookup:
                return None
            mesh_ids = supp_mesh_id_heading_mesh_id_lookup[mesh_id]
            for mesh_id in mesh_ids:
                tree_numbers.extend(desc_mesh_id_tree_number_lookup[mesh_id])
        elif mesh_id[0] == 'D':
            tree_numbers.extend(desc_mesh_id_tree_number_lookup[mesh_id])
        else:
            raise ValueError()
        return tree_numbers

    df_chemicals['MESH_tree'] = df_chemicals['MESH'].apply(lambda x: _get_mesh_tree(x))
    df_chemicals = df_chemicals.explode('MESH_tree')
    df_chemicals['group'] = df_chemicals['MESH_tree'].apply(
        lambda x: 'Unknown' if x is None else
        desc_tree_number_name_lookup[x.split('.')[0]]
    )
    df_chemicals['subgroup'] = df_chemicals['MESH_tree'].apply(
        lambda x: 'Unknown' if x is None else
        desc_tree_number_name_lookup['.'.join(x.split('.')[:2])]
    )

    chemical_entities = set(df_kg_contains['tail'].tolist())
    df_chemicals['is_contains'] = df_chemicals['foodatlas_id'].apply(
        lambda x: True if x in chemical_entities else False
    )
    df_chemicals = df_chemicals[['foodatlas_id', 'is_contains', 'group', 'subgroup']]
    df_chemicals.to_csv(
        os.path.join(args.output_dir, CHEMICALS_GROUP_FILENAME),
        sep='\t', index=False,
    )

    # organisms group and subgroup
    print('Getting organism group and subgroup...')
    df_foodb_foods = pd.read_csv(FOODB_FOOD_FILEPATH, keep_default_na=False)
    df_foodb_foods = df_foodb_foods[df_foodb_foods['ncbi_taxonomy_id'] != '']
    tax_id_group_mapping = dict(zip(
        df_foodb_foods['ncbi_taxonomy_id'], df_foodb_foods['food_group']))
    tax_id_subgroup_mapping = dict(zip(
        df_foodb_foods['ncbi_taxonomy_id'], df_foodb_foods['food_subgroup']))

    df_organisms = fa_kg.get_entities_by_type(startswith_type='organism')
    df_organisms['NCBI_taxonomy'] = df_organisms['other_db_ids'].apply(
        lambda x: x['NCBI_taxonomy'])
    df_organisms = df_organisms.explode('NCBI_taxonomy')

    df_organisms['group'] = df_organisms['NCBI_taxonomy'].apply(
        lambda x: 'Unknown' if x not in tax_id_group_mapping else tax_id_group_mapping[x]
    )
    df_organisms['subgroup'] = df_organisms['NCBI_taxonomy'].apply(
        lambda x: 'Unknown' if x not in tax_id_subgroup_mapping else tax_id_subgroup_mapping[x]
    )

    food_entities = set(df_kg_contains['head'].tolist())
    df_organisms['is_food'] = df_organisms['foodatlas_id'].apply(
        lambda x: True if x in food_entities else False
    )
    df_organisms = df_organisms[['foodatlas_id', 'is_food', 'group', 'subgroup']]
    df_organisms.to_csv(
        os.path.join(args.output_dir, ORGANISMS_GROUP_FILENAME),
        sep='\t', index=False,
    )

    print("Updating entities...")
    df_scientific_names = df_names[df_names['name_class'] == 'scientific name']
    ncbi_taxid_name_lookup = dict(zip(
        df_scientific_names['tax_id'], df_scientific_names['name_txt']))

    unknown_ncbi_taxids = []
    df_entities = fa_kg.get_all_entities()
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
