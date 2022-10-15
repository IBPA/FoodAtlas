import argparse
from glob import glob
from pathlib import Path
import random
import sys

sys.path.append('..')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PH_PAIRS_FILEPATH = "../../outputs/data_generation/ph_pairs_*.txt"
VAL_NUM_PREMISE = 500
TEST_NUM_PREMISE = 500
PER_ROUND_NUM_PREMISE = 500
PER_ROUND_NUM_PH_PAIRS = 1000
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
        "--round",
        type=int,
        required=True,
        help="What pre_annotation round are we generating.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Set if overwriting the output files.",
    )

    parser.add_argument(
        "--pre_annotation_filepath",
        type=str,
        required=True,
        help="Pre-annotation filepath.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Set random state.",
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
        "--per_round_num_premise",
        type=int,
        default=PER_ROUND_NUM_PREMISE,
        help=f"Number of premises to use per round. (Default: {PER_ROUND_NUM_PREMISE})",
    )

    parser.add_argument(
        "--per_round_num_ph_pairs",
        type=int,
        default=PER_ROUND_NUM_PH_PAIRS,
        help=f"Number of PH pairs to use per round. (Default: {PER_ROUND_NUM_PH_PAIRS})",
    )

    parser.add_argument(
        "--test_pre_annotation_filepath",
        type=str,
        default=TEST_PRE_ANNOTATION_FILEPATH,
        help=f"Test set pre-annotation filepath. (Default: {TEST_PRE_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--to_predict_filepath",
        type=str,
        required=True,
        help="To predict filepath.",
    )

    parser.add_argument(
        "--predicted_filepath",
        type=str,
        help="Predicted (by entailment model) filepath.",
    )

    parser.add_argument(
        "--skip_augment",
        action="store_true",
        help="Set if skipping augmentation.",
    )

    parser.add_argument(
        "--sampling_strategy",
        type=str,
        help="Sampling strategy to use.",
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

    Path(args.pre_annotation_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(args.to_predict_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_pre_annotation_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(args.test_pre_annotation_filepath).parent.mkdir(parents=True, exist_ok=True)

    if args.round == 1:
        df_ph_pairs = read_ph_pairs(args.ph_pairs_filepath)
        premises = list(set(df_ph_pairs["premise"].tolist()))
        random.shuffle(premises)
        val_premises = premises[0:args.val_num_premise]
        test_premises = premises[args.val_num_premise:args.val_num_premise+args.test_num_premise]
        remaining_premises = premises[args.val_num_premise+args.test_num_premise:]

        df_remaining = df_ph_pairs[df_ph_pairs["premise"].apply(lambda x: x in remaining_premises)]
        df_val = df_ph_pairs[df_ph_pairs["premise"].apply(lambda x: x in val_premises)]
        df_test = df_ph_pairs[df_ph_pairs["premise"].apply(lambda x: x in test_premises)]

        val_test_hypotheses = list(set(
            df_val["hypothesis_string"].tolist() + df_test["hypothesis_string"].tolist()))
        df_remaining = df_remaining[df_remaining["hypothesis_string"].apply(
            lambda x: x not in val_test_hypotheses)]

        print(f"df_ph_pairs shape: {df_ph_pairs.shape}")
        print(f"df_val shape: {df_val.shape}")
        print(f"df_test shape: {df_test.shape}")

        df_val.to_csv(args.val_pre_annotation_filepath, sep='\t', index=False)
        df_test.to_csv(args.test_pre_annotation_filepath, sep='\t', index=False)

        this_round_premises = random.sample(remaining_premises, args.per_round_num_premise)
        df_to_annotate = df_remaining[df_remaining["premise"].apply(
            lambda x: x in this_round_premises)].copy()

        df_to_annotate["source"] = f"annotation:round_{args.round}"
        print(f"Saving pre_annotation data to {args.pre_annotation_filepath}")
        df_to_annotate.to_csv(args.pre_annotation_filepath, sep='\t', index=False)

        df_to_predict = df_remaining[df_remaining["premise"].apply(
            lambda x: x not in this_round_premises)]

        print(f"Saving to_predict to {args.to_predict_filepath}")
        df_to_predict.to_csv(args.to_predict_filepath, sep='\t', index=False)
    else:
        df_predicted = pd.read_csv(args.predicted_filepath, sep='\t', keep_default_na=False)
        df_predicted.sort_values("prob", ascending=False, inplace=True)

        if args.sampling_strategy == "certain_pos":
            df_to_annotate = df_predicted[:args.per_round_num_ph_pairs].copy()
            df_to_predict = df_predicted[args.per_round_num_ph_pairs:].copy()
        elif args.sampling_strategy == "uncertain":
            df_predicted["uncertainty"] = df_predicted["prob"].apply(
                lambda x: np.min([1-x, x]))
            df_predicted.sort_values("uncertainty", ascending=False, inplace=True)

            df_to_annotate = df_predicted[:args.per_round_num_ph_pairs].copy()
            df_to_predict = df_predicted[args.per_round_num_ph_pairs:].copy()

            df_to_annotate.drop("uncertainty", axis=1, inplace=True)
            df_to_predict.drop("uncertainty", axis=1, inplace=True)
        elif args.sampling_strategy == "random_sample_each_bin":
            num_each_bin = int(args.per_round_num_ph_pairs/10)
            print(f"Number of samples in each bin: {num_each_bin}")

            to_annotate = []
            to_predict = []
            for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                df_subset = df_predicted[df_predicted["prob"] <= x]
                df_predicted = df_predicted[df_predicted["prob"] > x]
                df_subset = df_subset.sample(frac=1)
                to_annotate.append(df_subset[:num_each_bin].copy())
                to_predict.append(df_subset[num_each_bin:].copy())

            df_to_annotate = pd.concat(to_annotate)
            df_to_predict = pd.concat(to_predict)
        else:
            raise ValueError()

        df_to_annotate.drop("prob", axis=1, inplace=True)
        df_to_predict.drop("prob", axis=1, inplace=True)
        df_to_annotate["source"] = f"annotation:round_{args.round}"

        print(f"Saving pre_annotation data to {args.pre_annotation_filepath}")
        df_to_annotate.to_csv(args.pre_annotation_filepath, sep='\t', index=False)

        print(f"Saving to_predict to {args.to_predict_filepath}")
        df_to_predict.to_csv(args.to_predict_filepath, sep='\t', index=False)


if __name__ == '__main__':
    main()
