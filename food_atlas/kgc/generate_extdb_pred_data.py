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
EXTDB = ['Phenol-Explorer', 'FDC', 'Frida']


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
        default='../../outputs/kgc/data/production',
        help="Input full FAKG KGC directory.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='../../outputs/kgc/data/predict_extdb',
        help="Output directory.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    fa_kg = KnowledgeGraph(kg_dir=args.input_kg_dir)
    df_evidence = fa_kg.get_evidence()
    df_evidence_extdb = df_evidence[df_evidence['source'].apply(lambda x: x in EXTDB)]
    extdb_triples = set(df_evidence_extdb['triple'].tolist())

    # remove ExtDB triples
    for filename in ['train.txt', 'val_corrupted.txt', 'val_corrupted_no_label.txt']:
        if filename in ['train.txt', 'val_corrupted_no_label.txt']:
            names = ['head', 'relation', 'tail']
        elif filename == 'val_corrupted.txt':
            names = ['head', 'relation', 'tail', 'label']
        else:
            raise KeyError()

        df = pd.read_csv(
            os.path.join(args.input_kgc_dir, filename),
            sep='\t', names=names,
        )
        df['triple'] = df.apply(
            lambda row: f"({row['head']},{row['relation']},{row['tail']})", axis=1
        )
        print(f'Original df: {df.shape[0]}')

        df = df[df['triple'].apply(lambda x: x not in extdb_triples)]
        print(f'After dropping ExtDB df: {df.shape[0]}')

        df = df[names]
        df.to_csv(
            os.path.join(args.output_dir, filename),
            sep='\t', index=False, header=False
        )

    # test data
    df_evidence_extdb = df_evidence_extdb[['head', 'relation', 'tail']]
    df_evidence_extdb.drop_duplicates(inplace=True)
    df_evidence_extdb.to_csv(
            os.path.join(args.output_dir, 'test.txt'),
            sep='\t', index=False, header=False
        )


if __name__ == '__main__':
    main()
