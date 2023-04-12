import argparse
from itertools import product
import os
from pathlib import Path
import sys
import shutil

sys.path.append('../data_processing/')

import pandas as pd  # noqa: E402
from pandarallel import pandarallel  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402


PRODUCTION_DIR = 'production'
NAMES = ['head', 'relation', 'tail']


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hypotheses.")

    parser.add_argument(
        "--full_kg_dir",
        type=str,
        # required=True,
        default='../../outputs/kg/annotations_predictions_extdb_mesh_ncbi',
        help="Directory containing the full input KG.",
    )

    parser.add_argument(
        "--train_kgc_dir",
        type=str,
        # required=True,
        default='../../outputs/kgc/data/annotations_predictions-above80_extdb_mesh_ncbi',
        help="Directory containing the training data for KGC.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        # required=True,
        default='../../outputs/kgc/data/production',
        help="Output directory to save the production train/val data.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()
    pandarallel.initialize(progress_bar=True)

    # generate production data
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    shutil.copy(
        os.path.join(args.train_kgc_dir, 'train.txt'),
        os.path.join(args.output_dir, 'train.txt'),
    )

    df_val = pd.read_csv(
        os.path.join(args.train_kgc_dir, '../val_corrupted_no_label.txt'), sep='\t', header=None)
    df_test = pd.read_csv(
        os.path.join(args.train_kgc_dir, '../test_corrupted_no_label.txt'), sep='\t', header=None)
    df_new_val = pd.concat([df_val, df_test])
    # df_new_val.to_csv(
    #     os.path.join(args.output_dir, 'val_corrupted_no_label.txt'),
    #     sep='\t', index=False, header=False)

    df_val = pd.read_csv(
        os.path.join(args.train_kgc_dir, '../val_corrupted.txt'), sep='\t', header=None)
    df_test = pd.read_csv(
        os.path.join(args.train_kgc_dir, '../test_corrupted.txt'), sep='\t', header=None)
    df_new_val = pd.concat([df_val, df_test])
    # df_new_val.to_csv(
    #     os.path.join(args.output_dir, 'val_corrupted.txt'),
    #     sep='\t', index=False, header=False,
    # )

    # generate hypotheses
    fa_kg = KnowledgeGraph(kg_dir=args.full_kg_dir)

    df_relations = fa_kg.get_all_relations()
    contains_foodatlas_id = df_relations[
        df_relations["name"] == "contains"]["foodatlas_id"].tolist()[0]

    df_organism = fa_kg.get_entities_by_type(
        exact_type='organism', startswith_type='organism:')

    organism_foodatlas_ids = list(set(df_organism['foodatlas_id'].tolist()))

    #
    df_kg = fa_kg.get_kg()
    print(f'KG size: {df_kg.shape[0]}')

    df_contains = df_kg[df_kg['relation'] == contains_foodatlas_id]
    print(f'Size of contains triples: {df_contains.shape[0]}')

    df_organism_contains = df_contains[
        df_contains['head'].apply(lambda x: x in organism_foodatlas_ids)]
    print(f'Size of organism contains triples: {df_organism_contains.shape[0]}')

    # do negative sampling for the validation set
    head_pool = sorted(set(df_organism_contains['head'].tolist()))
    tail_pool = sorted(set(df_contains['tail'].tolist()))
    print(f'Head pool size: {len(head_pool)}')
    print(f'Tail pool size: {len(tail_pool)}')

    all_known_triples = df_kg.apply(
        lambda row: f"({row['head']}, {row['relation']}, {row['tail']})", axis=1).tolist()

    # generate hypotheses
    hypotheses = list(product(head_pool, [contains_foodatlas_id], tail_pool))
    df_hypotheses = pd.DataFrame(hypotheses, columns=['head', 'relation', 'tail'])
    print(f'Size of hypotheses: {df_hypotheses.shape[0]}')

    def is_known(row):
        if f"({row['head']}, {row['relation']}, {row['tail']})" in all_known_triples:
            return True
        else:
            return False

    df_hypotheses['is_known'] = df_hypotheses.parallel_apply(lambda row: is_known(row), axis=1)

    df_hypotheses = df_hypotheses[~df_hypotheses['is_known']]
    print(f'Hypotheses size after dropping known triples: {df_hypotheses.shape[0]}')
    df_hypotheses = df_hypotheses[['head', 'relation', 'tail']]
    # df_hypotheses.to_csv(
    #     os.path.join(args.output_dir, 'test.txt'),
    #     sep='\t', index=False, header=False
    # )


if __name__ == '__main__':
    main()
