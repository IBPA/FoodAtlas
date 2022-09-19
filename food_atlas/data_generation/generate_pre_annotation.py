import argparse
from glob import glob
from pathlib import Path
import random
import sys

sys.path.append('..')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PH_PAIRS_FILEPATH = "../../outputs/data_generation/ph_pairs_*.txt"
PRE_ANNOTATION_FILEPATH = "../../outputs/data_generation/pre_annotation_*.tsv"
VAL_NUM_PREMISE = 500
TEST_NUM_PREMISE = 500
PER_ROUND_NUM_PREMISE = 500
VAL_PRE_ANNOTATION_FILEPATH = "../../outputs/data_generation/val_pre_annotation.tsv"
TEST_PRE_ANNOTATION_FILEPATH = "../../outputs/data_generation/test_pre_annotation.tsv"
TO_PREDICT_FILEPATH = "../../outputs/data_generation/to_predict_*.tsv"
PREDICTED_FILEPATH = "../../outputs/data_generation/predicted_*.tsv"
SAMPLING_REASON_FILEPATH = "../../outputs/data_generation/sampling_reason_*.tsv"


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
        default=PRE_ANNOTATION_FILEPATH,
        help=f"Annotation filepath. (Default: {PRE_ANNOTATION_FILEPATH})",
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
        "--test_pre_annotation_filepath",
        type=str,
        default=TEST_PRE_ANNOTATION_FILEPATH,
        help=f"Test set pre-annotation filepath. (Default: {TEST_PRE_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--to_predict_filepath",
        type=str,
        default=TO_PREDICT_FILEPATH,
        help=f"To predict filepath. (Default: {TO_PREDICT_FILEPATH})",
    )

    parser.add_argument(
        "--predicted_filepath",
        type=str,
        default=PREDICTED_FILEPATH,
        help=f"Predicted (by entailment model) filepath. (Default: {PREDICTED_FILEPATH})",
    )

    parser.add_argument(
        "--sampling_reason_filepath",
        type=str,
        default=SAMPLING_REASON_FILEPATH,
        help=f"Save how we selected the premises through sampling. (Default: {SAMPLING_REASON_FILEPATH})",
    )

    parser.add_argument(
        "--skip_augment",
        action="store_true",
        help="Set if skipping augmentation.",
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

    pre_annotation_filepath = args.pre_annotation_filepath.replace('*', str(args.round))
    if Path(pre_annotation_filepath).is_file() and args.overwrite is False:
        raise RuntimeError(f"Pre-annotation file for round {args.round} already exists!")

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

        print(f"df_ph_pairs shape: {df_ph_pairs.shape}")
        print(f"df_val shape: {df_val.shape}")
        print(f"df_test shape: {df_test.shape}")

        df_val.to_csv(args.val_pre_annotation_filepath, sep='\t', index=False)
        df_test.to_csv(args.test_pre_annotation_filepath, sep='\t', index=False)

        this_round_premises = random.sample(remaining_premises, args.per_round_num_premise)
        df_to_annotate = df_remaining[df_remaining["premise"].apply(
            lambda x: x in this_round_premises)].copy()

        df_to_predict = df_remaining[df_remaining["premise"].apply(
            lambda x: x not in this_round_premises)]
    else:
        predicted_filepath = args.predicted_filepath.replace('*', str(args.round - 1))
        df_predicted = pd.read_csv(predicted_filepath, sep='\t', keep_default_na=False)
        df_predicted_orig = df_predicted.copy()
        df_predicted["uncertainty"] = df_predicted["prob"].apply(lambda x: min(x, 1-x))

        df_predicted["entailment_rank"] = df_predicted["prob"].rank(ascending=False)
        df_predicted["uncertainty_rank"] = df_predicted["uncertainty"].rank(ascending=False)

        df_grouped_by_premise = df_predicted.\
            groupby("premise")[["entailment_rank", "uncertainty_rank"]].\
            agg(lambda x: np.mean(list(x))).reset_index()

        df_grouped_by_premise["coin_flip"] = [
            random.random() for _ in range(df_grouped_by_premise.shape[0])]
        df_grouped_by_premise["representative_rank"] = df_grouped_by_premise.apply(
            lambda row: row["entailment_rank"] if row["coin_flip"] < 0.5
            else row["uncertainty_rank"],
            axis=1)
        df_grouped_by_premise["certain_or_uncertain"] = \
            df_grouped_by_premise["coin_flip"].apply(
                lambda x: "certain" if x < 0.5 else "uncertain")
        df_grouped_by_premise.sort_values("representative_rank", inplace=True)

        sampling_reason_filepath = args.sampling_reason_filepath.replace('*', str(args.round - 1))
        if Path(sampling_reason_filepath).is_file() and args.overwrite is False:
            raise RuntimeError("Sampling reason already exists!")
        print(f"Saving sampling reason to {sampling_reason_filepath}")
        df_grouped_by_premise.to_csv(sampling_reason_filepath, sep='\t', index=False)

        selected_premises = df_grouped_by_premise.head(
            args.per_round_num_premise)["premise"].tolist()

        df_grouped_by_premise.index = df_grouped_by_premise["premise"]

        df_to_annotate = df_predicted_orig[df_predicted_orig["premise"].apply(
            lambda x: x in selected_premises)].copy()
        df_to_annotate["certain_or_uncertain"] = df_to_annotate["premise"].apply(
            lambda x: df_grouped_by_premise.at[x, "certain_or_uncertain"])

        df_to_predict = df_predicted_orig[df_predicted_orig["premise"].apply(
            lambda x: x not in selected_premises)].copy()

    df_to_annotate["round"] = args.round
    print(f"Saving pre_annotation data to {pre_annotation_filepath}")
    df_to_annotate.to_csv(pre_annotation_filepath, sep='\t', index=False)

    to_predict_filepath = args.to_predict_filepath.replace('*', str(args.round))
    if Path(to_predict_filepath).is_file() and args.overwrite is False:
        raise RuntimeError(f"To-predict file for round {args.round} already exists!")
    print(f"Saving to_predict to {to_predict_filepath}")
    df_to_predict.to_csv(to_predict_filepath, sep='\t', index=False)


if __name__ == '__main__':
    main()
