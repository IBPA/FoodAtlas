import argparse
from glob import glob
from pathlib import Path
import random
import sys

sys.path.append('..')

import pandas as pd  # noqa: E402


PH_PAIRS_FILEPATH = "../../outputs/data_generation/ph_pairs_*.txt"
VAL_NUM_PREMISE = 500
TEST_NUM_PREMISE = 500
TRAIN_NUM_PH_PAIRS = 5000
VAL_PRE_ANNOTATION_FILEPATH = "../../outputs/data_generation/val_pre_annotation.tsv"
TEST_PRE_ANNOTATION_FILEPATH = "../../outputs/data_generation/test_pre_annotation.tsv"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--ph_pairs_filepath",
        type=str,
        default=PH_PAIRS_FILEPATH,
        help=f"PH pairs filepath. (Default: {PH_PAIRS_FILEPATH})",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Set random state.",
    )

    parser.add_argument(
        "--train_num_ph_pairs",
        type=int,
        default=TRAIN_NUM_PH_PAIRS,
        help=f"Number of PH pairs to annotate. (Default: {TRAIN_NUM_PH_PAIRS})",
    )

    parser.add_argument(
        "--train_pre_annotation_filepath",
        type=str,
        required=True,
        help="Pre-annotation filepath.",
    )

    parser.add_argument(
        "--val_num_premise",
        type=int,
        default=VAL_NUM_PREMISE,
        help=f"Number of premises to use for validation. (Default: {VAL_NUM_PREMISE})",
    )

    parser.add_argument(
        "--val_pre_annotation_filepath",
        type=str,
        default=VAL_PRE_ANNOTATION_FILEPATH,
        help=f"Validation pre-annotation set filepath. (Default: {VAL_PRE_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--test_num_premise",
        type=int,
        default=TEST_NUM_PREMISE,
        help=f"Number of premises to use for test. (Default: {TEST_NUM_PREMISE})",
    )

    parser.add_argument(
        "--test_pre_annotation_filepath",
        type=str,
        default=TEST_PRE_ANNOTATION_FILEPATH,
        help=f"Test set pre-annotation filepath. (Default: {TEST_PRE_ANNOTATION_FILEPATH})",
    )

    args = parser.parse_args()
    return args


def read_ph_pairs(ph_pairs_filepath: str) -> pd.DataFrame:
    if '*' in ph_pairs_filepath:
        ph_pairs_filepath = sorted(glob(ph_pairs_filepath))[-1]
        print(f"Using the latest time-stamped PH pairs: {ph_pairs_filepath}")
    else:
        print(f"Using the user-specified PH pairs: {ph_pairs_filepath}")

    df = pd.read_csv(ph_pairs_filepath, sep='\t', keep_default_na=False)
    print(f"Input PH pairs dataframe shape: {df.shape}")

    return df


def main():
    args = parse_argument()

    if args.random_state:
        random.seed(args.random_state)

    Path(args.train_pre_annotation_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_pre_annotation_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(args.test_pre_annotation_filepath).parent.mkdir(parents=True, exist_ok=True)

    df_ph_pairs = read_ph_pairs(args.ph_pairs_filepath)
    premises = list(set(df_ph_pairs["premise"].tolist()))
    random.shuffle(premises)
    val_premises = premises[0:args.val_num_premise]
    test_premises = premises[args.val_num_premise:args.val_num_premise+args.test_num_premise]
    remaining_premises = premises[args.val_num_premise+args.test_num_premise:]

    df_remaining = df_ph_pairs[df_ph_pairs["premise"].apply(lambda x: x in remaining_premises)]
    df_val = df_ph_pairs[df_ph_pairs["premise"].apply(lambda x: x in val_premises)]
    df_test = df_ph_pairs[df_ph_pairs["premise"].apply(lambda x: x in test_premises)]

    val_hyp = list(set(df_val["hypothesis_string"].tolist()))
    test_hyp = list(set(df_test["hypothesis_string"].tolist()))
    val_test_hyp = list(set(val_hyp + test_hyp))

    df_remaining = df_remaining[df_remaining["hypothesis_string"].apply(
        lambda x: x not in val_test_hyp)]
    df_val = df_val[df_val["hypothesis_string"].apply(
        lambda x: x not in test_hyp)]

    print(f"df_ph_pairs shape: {df_ph_pairs.shape}")
    print(f"df_remaining shape: {df_remaining.shape}")
    print(f"df_val shape: {df_val.shape}")
    print(f"df_test shape: {df_test.shape}")

    df_val["source"] = "annotation:val"
    df_test["source"] = "annotation:test"
    df_val.to_csv(args.val_pre_annotation_filepath, sep='\t', index=False)
    df_test.to_csv(args.test_pre_annotation_filepath, sep='\t', index=False)

    df_train = df_remaining.sample(n=args.train_num_ph_pairs, random_state=args.random_state)
    df_train["source"] = "annotation:train"
    print(f"Saving train pre_annotation data to {args.train_pre_annotation_filepath}")
    df_train.to_csv(args.train_pre_annotation_filepath, sep='\t', index=False)


if __name__ == '__main__':
    main()
