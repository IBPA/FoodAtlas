import argparse
import os
from pathlib import Path
import pickle
import sys

from tqdm import tqdm
import pandas as pd


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file.',
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file.',
    )

    args = parser.parse_args()
    return args


def parse_input(filepath):
    with open(filepath, 'rb') as _f:
        df = pickle.load(_f)

    df.reset_index(inplace=True, drop=True)

    return df


def main():
    args = parse_argument()

    df = parse_input(args.input)
    df = df.sample(frac=1, ignore_index=True)

    rows = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        annotations = row['annotations']
        if annotations is None:
            continue

        species_exist = False
        chemical_exist = False
        for entity in annotations:
            if '|species|' in entity:
                species_exist = True
            if '|chemical|' in entity:
                chemical_exist = True

            if species_exist and chemical_exist:
                rows.append(row)
                break

    df_output = pd.DataFrame(rows)
    df_output = df_output.head(100)

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    with open(args.output, 'wb') as _f:
        pickle.dump(df_output, _f)


if __name__ == '__main__':
    main()
