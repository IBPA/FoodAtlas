import argparse
from itertools import product
from collections import Counter
import os
from pathlib import Path
import random
import shutil
import sys

sys.path.append('../data_processing/')

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402
from common_utils.utils import load_pkl  # noqa: E402
from common_utils.chemical_db_ids import (
    _get_name_from_json, _get_summary_description_from_json,
    read_mesh_data, get_mesh_name_using_mesh_id
)  # noqa: E402
from merge_ncbi_taxonomy import read_dmp_files  # noqa: E402

ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"
TRAIN_FILENAME = "train.txt"
VAL_FILENAME = "val.txt"
VAL_CORRUPTED_FILENAME = "val_corrupted.txt"
VAL_CORRUPTED_NO_LABEL_FILENAME = "val_corrupted_no_label.txt"
TEST_FILENAME = "test.txt"
TEST_CORRUPTED_FILENAME = "test_corrupted.txt"
TEST_CORRUPTED_NO_LABEL_FILENAME = "test_corrupted_no_label.txt"
CID_JSON_LOOKUP_PKL_FILEPATH = "../../data/FoodAtlas/cid_json_lookup.pkl"
NCBI_NAME_CLASS_TO_USE = ["genbank common name", "scientific name", "common name"]
NCBI_NAMES_FILEPATH = "../../data/NCBI_Taxonomy/names.dmp"
PARTS_FILEPATH = "../../data/FoodAtlas/food_parts.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the train/val/test data for KGC.")

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        required=True,
        help="Input KG directory.",
    )

    parser.add_argument(
        "--full_kg_dir",
        type=str,
        help="KG directory containing the complete data.",
    )

    parser.add_argument(
        "--val_test_dir",
        type=str,
        required=True,
        help="Output directory to save/load val/test set.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save train related data.",
    )

    parser.add_argument(
        "--data_split",
        type=str,
        default='70:15:15',
        help='train:val:test split expressed in ratio (Default: 70:15:15)',
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=530,
        help='Random state (Default: 530)',
    )

    parser.add_argument(
        "--is_initial",
        action='store_true',
        help='Is this initial data generation? If set, generate val/test set.',
    )

    args = parser.parse_args()

    if args.is_initial:
        assert args.full_kg_dir is not None
        assert args.data_split is not None

    return args


def _check_overlap(df_a, df_b):
    def _get_triple_str(row):
        return f"{row['head']},{row['relation']},{row['tail']}"

    a_triples = df_a.apply(lambda row: _get_triple_str(row), axis=1).tolist()
    b_triples = df_b.apply(lambda row: _get_triple_str(row), axis=1).tolist()
    overlap = set(a_triples) & set(b_triples)
    if len(overlap) > 0:
        raise RuntimeError()


def main():
    args = parse_argument()

    random.seed(args.random_state)

    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir)

    df_relations = fa_kg.get_all_relations()
    contains_foodatlas_id = df_relations[
        df_relations["name"] == "contains"]["foodatlas_id"].tolist()[0]

    df_organism = fa_kg.get_entities_by_type(
        exact_type='organism', startswith_type='organism:')
    df_organism_with_part = fa_kg.get_entities_by_type(
        exact_type='organism_with_part', startswith_type='organism_with_part:')

    organism_foodatlas_ids = df_organism['foodatlas_id'].tolist()
    organism_with_part_foodatlas_ids = df_organism_with_part['foodatlas_id'].tolist()

    #
    df_kg = fa_kg.get_kg()
    print(f'KG size: {df_kg.shape[0]}')

    df_contains = df_kg[df_kg['relation'] == contains_foodatlas_id]
    print(f'Size of contains triples: {df_contains.shape[0]}')

    df_organism_contains = df_contains[
        df_contains['head'].apply(lambda x: x in organism_foodatlas_ids)]
    print(f'Size of organism contains triples: {df_organism_contains.shape[0]}')

    df_organism_with_part_contains = df_contains[
        df_contains['head'].apply(lambda x: x in organism_with_part_foodatlas_ids)]
    print(f'Size of organism_with_part contains triples: {df_organism_with_part_contains.shape[0]}')

    df_others = df_kg[df_kg['relation'] != contains_foodatlas_id]
    print(f'Size of non-contains triples: {df_others.shape[0]}')

    # shuffle
    df_organism_contains = df_organism_contains.sample(
        frac=1, random_state=args.random_state)
    df_organism_with_part_contains = df_organism_with_part_contains.sample(
        frac=1, random_state=args.random_state)
    df_others = df_others.sample(
        frac=1, random_state=args.random_state)

    #
    if args.is_initial:
        num_train = int(df_organism_contains.shape[0] * int(args.data_split.split(':')[0]) / 100)
        num_val = int(df_organism_contains.shape[0] * int(args.data_split.split(':')[1]) / 100)
        df_organism_contains_train = df_organism_contains[:num_train]
        df_organism_contains_val = df_organism_contains[num_train:num_train+num_val]
        df_organism_contains_test = df_organism_contains[num_train+num_val:]
    else:
        df_organism_contains_train = df_organism_contains
        df_organism_contains_val = pd.read_csv(
            os.path.join(args.val_test_dir, VAL_FILENAME),
            sep='\t', names=['head', 'relation', 'tail'],
        )
        df_organism_contains_test = pd.read_csv(
            os.path.join(args.val_test_dir, TEST_FILENAME),
            sep='\t', names=['head', 'relation', 'tail'],
        )

        # drop val/test from train
        df_organism_contains_train = pd.concat([
            df_organism_contains_train,
            df_organism_contains_val,
            df_organism_contains_test,
        ])
        df_organism_contains_train.drop_duplicates(inplace=True, keep=False)

    # exclude these NCBI taxonomies since they're in the val/test set
    organism_val_test_entities = \
        df_organism_contains_val['head'].tolist() + df_organism_contains_test['head'].tolist()
    organism_val_test_entities = list(set(organism_val_test_entities))
    ncbi_taxonomies_to_exclude = []
    for x in organism_val_test_entities:
        entity = fa_kg.get_entity_by_id(x)
        ncbi_taxonomies_to_exclude.extend(entity['other_db_ids']['NCBI_taxonomy'])
    ncbi_taxonomies_to_exclude = list(set(ncbi_taxonomies_to_exclude))

    # do the exclusion
    def _f(head):
        entity = fa_kg.get_entity_by_id(head)
        if entity['other_db_ids']['NCBI_taxonomy'][0] in ncbi_taxonomies_to_exclude:
            return False
        else:
            return True

    df_organism_with_part_contains = df_organism_with_part_contains[
        df_organism_with_part_contains['head'].apply(lambda x: _f(x))
    ]

    df_train = pd.concat([df_organism_contains_train, df_organism_with_part_contains, df_others])
    print(f'Train size: {df_train.shape[0]}')

    _check_overlap(df_train, df_organism_contains_val)
    _check_overlap(df_train, df_organism_contains_test)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print('Saving training data...')
    df_train.to_csv(
        os.path.join(args.output_dir, TRAIN_FILENAME), sep='\t', index=False, header=False)

    if args.is_initial:
        df_val = df_organism_contains_val
        df_test = df_organism_contains_test

        print(f'Val size: {df_val.shape[0]}')
        print(f'Test size: {df_test.shape[0]}')

        _check_overlap(df_organism_contains_val, df_organism_contains_test)

        print('Saving val data...')
        df_val.to_csv(
            os.path.join(args.val_test_dir, VAL_FILENAME), sep='\t', index=False, header=False)

        print('Saving test data...')
        df_test.to_csv(
            os.path.join(args.val_test_dir, TEST_FILENAME), sep='\t', index=False, header=False)

        # do negative sampling for the validation set
        head_pool = sorted(set(df_organism_contains['head'].tolist()))
        tail_pool = sorted(set(df_contains['tail'].tolist()))

        fa_full_kg = KnowledgeGraph(kg_dir=args.full_kg_dir)
        df_full_kg = fa_full_kg.get_kg()
        print(f'Size of full KG: {df_full_kg.shape[0]}')

        all_known_triples = df_full_kg.apply(
            lambda row: f"({row['head']}, {row['relation']}, {row['tail']})", axis=1).tolist()

        df_val_corrupted = df_val.copy()
        df_val_corrupted['label'] = 1

        def _generate_negatives(df):
            rows = []
            for _, row in tqdm(df.iterrows(), total=df.shape[0]):
                rows.append(row.to_dict())

                head_pool_copy = head_pool.copy()
                tail_pool_copy = tail_pool.copy()
                random.shuffle(head_pool_copy)
                random.shuffle(tail_pool_copy)

                # corrupt head
                sampled_head = None
                for e in head_pool_copy:
                    if f"({e}, {row['relation']}, {row['tail']})" not in all_known_triples:
                        sampled_head = e
                        break

                if sampled_head:
                    head_corrupt_triple = row.to_dict()
                    head_corrupt_triple['head'] = sampled_head
                    head_corrupt_triple['label'] = 0
                    rows.append(head_corrupt_triple)
                else:
                    raise RuntimeError()

                # corrupt tail
                sampled_tail = None
                for e in tail_pool_copy:
                    if f"({row['head']}, {row['relation']}, {e})" not in all_known_triples:
                        sampled_tail = e
                        break

                if sampled_tail:
                    tail_corrupt_triple = row.to_dict()
                    tail_corrupt_triple['tail'] = sampled_tail
                    tail_corrupt_triple['label'] = 0
                    rows.append(tail_corrupt_triple)
                else:
                    raise RuntimeError()

            return pd.DataFrame(rows)

        df_val_corrupted = _generate_negatives(df_val_corrupted)
        print(f'Corrupted val size: {df_val_corrupted.shape[0]}')

        df_val_corrupted.to_csv(
            os.path.join(args.val_test_dir, VAL_CORRUPTED_FILENAME),
            sep='\t',
            index=False,
            header=False,
        )

        df_val_corrupted_no_label = df_val_corrupted.drop('label', axis=1)
        df_val_corrupted_no_label.to_csv(
            os.path.join(args.val_test_dir, VAL_CORRUPTED_NO_LABEL_FILENAME),
            sep='\t',
            index=False,
            header=False,
        )

        # do negative sampling for the test set
        df_test_corrupted = df_test.copy()
        df_test_corrupted['label'] = 1

        df_test_corrupted = _generate_negatives(df_test_corrupted)
        print(f'Corrupted test size: {df_test_corrupted.shape[0]}')

        df_test_corrupted.to_csv(
            os.path.join(args.val_test_dir, TEST_CORRUPTED_FILENAME),
            sep='\t',
            index=False,
            header=False,
        )

        df_test_corrupted_no_label = df_test_corrupted.drop('label', axis=1)
        df_test_corrupted_no_label.to_csv(
            os.path.join(args.val_test_dir, TEST_CORRUPTED_NO_LABEL_FILENAME),
            sep='\t',
            index=False,
            header=False,
        )

    # copy other files
    df_entities = fa_kg.get_all_entities()

    print('Loading cid_json_lookup...')
    cid_json_lookup = load_pkl(CID_JSON_LOOKUP_PKL_FILEPATH)

    print('Loading mesh_data_dict...')
    mesh_data_dict = read_mesh_data()

    print('Loading NCBI names...')
    df_names = read_dmp_files(NCBI_NAMES_FILEPATH, filetype="names")
    df_names = df_names[df_names["name_class"].apply(lambda x: x in NCBI_NAME_CLASS_TO_USE)]
    df_names = df_names.groupby("tax_id")["name_txt"].apply(set).reset_index()
    names_lookup = dict(zip(
        df_names['tax_id'].tolist(), df_names['name_txt'].tolist()
    ))

    print('Loading food parts...')
    df_food_parts = pd.read_csv(PARTS_FILEPATH, sep='\t')
    food_parts_lookup = dict(zip(
        df_food_parts['foodatlas_part_id'].tolist(), df_food_parts['food_part'].tolist()
    ))

    def _get_short_name(row):
        other_db_ids = row['other_db_ids']
        if row['type'] == 'chemical':
            if 'PubChem' in other_db_ids:
                json = cid_json_lookup[other_db_ids['PubChem'][0]]
                return _get_name_from_json(json)
            elif 'MESH' in other_db_ids:
                mesh_id = other_db_ids['MESH'][0]
                return get_mesh_name_using_mesh_id(mesh_id, mesh_data_dict)
            else:
                raise ValueError()
        elif row['type'] == 'organism_with_part' or row['type'].startswith('organism_with_part:'):
            foodatlas_part_id = other_db_ids['foodatlas_part_id']
            part_name = food_parts_lookup[foodatlas_part_id]
            ncbi_taxonomy = other_db_ids['NCBI_taxonomy'][0]
            if ncbi_taxonomy in names_lookup:
                names = names_lookup[ncbi_taxonomy]
                names = [f'{x} {part_name}' for x in names]
                return ' '.join(names)
            else:
                return ' '.join(row['name'].split(' - '))
        elif row['type'] == 'organism' or row['type'].startswith('organism:'):
            ncbi_taxonomy = other_db_ids['NCBI_taxonomy'][0]
            if ncbi_taxonomy in names_lookup:
                names = names_lookup[ncbi_taxonomy]
                return ' '.join(names)
            else:
                return row['name']
        else:
            raise ValueError()

    def _get_long_name(row):
        other_db_ids = row['other_db_ids']
        if row['type'] == 'chemical':
            if 'PubChem' in other_db_ids:
                json = cid_json_lookup[other_db_ids['PubChem'][0]]
                descriptions = _get_summary_description_from_json(json)
                if descriptions:
                    return row['short_name'] + ' ' + descriptions[0]
                else:
                    return row['short_name']
            elif 'MESH' in other_db_ids:
                return row['short_name']
            else:
                raise ValueError()
        elif row['type'] == 'organism_with_part' or row['type'].startswith('organism_with_part:'):
            return row['short_name']
        elif row['type'] == 'organism' or row['type'].startswith('organism:'):
            return row['short_name']
        else:
            raise ValueError()

    print('Finding short names...')
    df_entities['short_name'] = df_entities.apply(lambda row: _get_short_name(row), axis=1)
    print('Finding long names...')
    df_entities['long_name'] = df_entities.apply(lambda row: _get_long_name(row), axis=1)
    df_entities.to_csv(os.path.join(args.output_dir, ENTITIES_FILENAME), sep='\t', index=False)

    short_names = df_entities['short_name'].tolist()
    if len(short_names) != len(set(short_names)):
        duplicates = [k for k, v in Counter(short_names).items() if v > 1]
        print(f'!!!Found duplicates in short names!!!: {duplicates}')

        for duplicate in duplicates:
            df_duplicates = df_entities[df_entities['short_name'] == duplicate]
            for idx, row in df_duplicates.iterrows():
                df_entities.at[idx, 'short_name'] = _get_long_name(row)

    # check again
    short_names = df_entities['short_name'].tolist()
    if len(short_names) != len(set(short_names)):
        duplicates = [k for k, v in Counter(short_names).items() if v > 1]
        print(f'Found duplicates again: {duplicates}')
        # these duplicates are ok

    long_names = df_entities['long_name'].tolist()
    if len(long_names) != len(set(long_names)):
        duplicates = [k for k, v in Counter(long_names).items() if v > 1]
        print(f'!!!Found duplicates in long names!!!: {duplicates}')

    shutil.copy(
        os.path.join(args.input_kg_dir, RELATIONS_FILENAME),
        os.path.join(args.output_dir, RELATIONS_FILENAME)
    )

    print('Finished generating data.')


if __name__ == '__main__':
    main()
