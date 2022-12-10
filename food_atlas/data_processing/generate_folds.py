import argparse
import os
from pathlib import Path
import random
import sys

import pandas as pd
from sklearn.model_selection import KFold

NUM_FOLDS = 10


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create folds for training/validating the deployment model.")

    parser.add_argument(
        "--input_train_filepath",
        type=str,
        required=True,
        help="Filepath of the train file."
    )

    parser.add_argument(
        "--input_val_filepath",
        type=str,
        required=True,
        help="Filepath of the val file."
    )

    parser.add_argument(
        "--input_test_filepath",
        type=str,
        required=True,
        help="Filepath of the test file."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the folds."
    )

    parser.add_argument(
        "--num_folds",
        type=int,
        default=NUM_FOLDS,
        help="Number of folds."
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    df_input_train = pd.read_csv(args.input_train_filepath, sep='\t', keep_default_na=False)
    df_input_val = pd.read_csv(args.input_val_filepath, sep='\t', keep_default_na=False)
    df_input_test = pd.read_csv(args.input_test_filepath, sep='\t', keep_default_na=False)

    print(f"Shape of train: {df_input_train.shape[0]}")
    print(f"Shape of val: {df_input_val.shape[0]}")
    print(f"Shape of test: {df_input_test.shape[0]}")

    df = pd.concat([df_input_train, df_input_val, df_input_test])
    print(f"Shape of all annotated data: {df.shape[0]}")

    premise = list(set(df["premise"].tolist()))
    print(f"Number of unique premises: {len(premise)}")

    std = 100
    while std > 26:
        kf = KFold(n_splits=args.num_folds, shuffle=True)
        train_shape = []
        val_shape = []
        for i, (train_idx, val_idx) in enumerate(kf.split(premise)):
            print(f"Processing fold: {i}")
            premise_train = [premise[x] for x in train_idx]
            premise_val = [premise[x] for x in val_idx]

            df_train = df[df["premise"].apply(lambda x: x in premise_train)]
            df_val = df[df["premise"].apply(lambda x: x in premise_val)]

            train_output_filepath = os.path.join(args.output_dir, f"fold_{i}", "train.tsv")
            val_output_filepath = os.path.join(args.output_dir, f"fold_{i}", "val.tsv")

            Path(train_output_filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(val_output_filepath).parent.mkdir(parents=True, exist_ok=True)

            df_train.to_csv(train_output_filepath, sep='\t', index=False)
            df_val.to_csv(val_output_filepath, sep='\t', index=False)

            train_shape.append(df_train.shape[0])
            val_shape.append(df_val.shape[0])

        print(train_shape)
        print(val_shape)

        import numpy as np
        std = np.std(val_shape)
        print(std)


if __name__ == "__main__":
    main()
