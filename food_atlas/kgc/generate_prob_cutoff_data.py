import argparse
import os
from pathlib import Path
import shutil
import sys

sys.path.append('../data_processing/')

import pandas as pd  # noqa: E402
from pandarallel import pandarallel  # noqa: E402

from common_utils.knowledge_graph import KnowledgeGraph  # noqa: E402

ENTITIES_FILENAME = "entities.txt"
RELATIONS_FILENAME = "relations.txt"
TRAIN_FILENAME = "train.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the train/val/test data for KGC.")

    parser.add_argument(
        "--input_kg_dir",
        type=str,
        default='../../outputs/kg/annotations_predictions_extdb_mesh_ncbi',
        help="Input full FAKG directory.",
    )

    parser.add_argument(
        "--input_kgc_dir",
        type=str,
        default='../../outputs/kgc/data/annotations_predictions_extdb_mesh_ncbi',
        help="Input full FAKG KGC directory.",
    )

    parser.add_argument(
        "--output_dir_fstr",
        type=str,
        default='../../outputs/kgc/data/annotations_predictions-above{}_extdb_mesh_ncbi',
        help="Output directory.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    pandarallel.initialize(progress_bar=True)

    df_train = pd.read_csv(
        os.path.join(args.input_kgc_dir, TRAIN_FILENAME),
        sep='\t', names=['head', 'relation', 'tail'],
    )
    df_train['triple'] = df_train.apply(
        lambda row: f"({row['head']},{row['relation']},{row['tail']})", axis=1
    )

    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir)
    df_evidence = fa_kg.get_evidence()
    df_evidence['prob_mean'] = df_evidence['prob_mean'].apply(lambda x: '1.0' if x == '' else x)
    df_evidence['prob_mean'] = df_evidence['prob_mean'].astype('float')

    for prob_cutoff in [0.6, 0.7, 0.8, 0.9]:
        print(f'Processing prob cutoff: {prob_cutoff}')
        output_dir = args.output_dir_fstr.format(int(prob_cutoff*100))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        df = df_evidence[df_evidence['prob_mean'] >= prob_cutoff]
        triples = list(set(df['triple'].tolist()))

        df_train_copy = df_train.copy()
        df_train_copy = df_train_copy[df_train_copy['triple'].parallel_apply(
            lambda x: x in triples)]
        df_train_copy = df_train_copy[['head', 'relation', 'tail']]
        df_train_copy.to_csv(
            os.path.join(output_dir, TRAIN_FILENAME),
            sep='\t', index=False, header=False
        )

        shutil.copy(
            os.path.join(args.input_kgc_dir, ENTITIES_FILENAME),
            os.path.join(output_dir, ENTITIES_FILENAME)
        )

        shutil.copy(
            os.path.join(args.input_kgc_dir, RELATIONS_FILENAME),
            os.path.join(output_dir, RELATIONS_FILENAME)
        )


if __name__ == '__main__':
    main()
