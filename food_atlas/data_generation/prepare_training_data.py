import argparse
import math
from pathlib import Path
import random
import sys

sys.path.append('..')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


TRAIN_POOL_FILEPATH = "../../outputs/data_generation/train_pool.tsv"
TRAIN_FILEPATH_FS = "../../outputs/data_generation/{}/run_{}/round_{}/train.tsv"
TO_PREDICT_FILEPATH_FS = "../../outputs/data_generation/{}/run_{}/round_{}/to_predict.tsv"
PREDICTED_FILEPATH_FS = "../../outputs/data_generation/{}/run_{}/round_{}/predicted.tsv"
STRATIFIED_NUM_BINS = 10


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--random_state",
        type=int,
        help="Set random state.",
    )

    parser.add_argument(
        "--sampling_strategy",
        type=str,
        required=True,
        help="Sampling strategy to use (stratified|certain_pos|uncertain).",
    )

    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="Specify which run this is for.",
    )

    parser.add_argument(
        "--round",
        type=int,
        required=True,
        help="Current round.",
    )

    parser.add_argument(
        "--total_rounds",
        type=int,
        required=True,
        help="Total number of rounds.",
    )

    parser.add_argument(
        "--train_pool_filepath",
        type=str,
        default=TRAIN_POOL_FILEPATH,
        help=f"Training pool filepath. (Default: {TRAIN_POOL_FILEPATH})",
    )

    args = parser.parse_args()
    return args


def _drop_duplicates():
    pass


def main():
    args = parse_argument()

    print(f"*****{args.sampling_strategy}|run_{args.run}|round_{args.round} START *****")

    if args.random_state:
        random.seed(args.random_state)

    df_train_pool = pd.read_csv(args.train_pool_filepath, sep='\t')
    df_train_pool = df_train_pool.sample(frac=1, random_state=args.random_state)
    print(f"Train pool shape: {df_train_pool.shape[0]}")

    num_train_per_round = math.ceil(df_train_pool.shape[0] / args.total_rounds)
    print(f"Number of training data per round: {num_train_per_round}")

    train_filepath = TRAIN_FILEPATH_FS.format(
        args.sampling_strategy, args.run, args.round)
    to_predict_filepath = TO_PREDICT_FILEPATH_FS.format(
        args.sampling_strategy, args.run, args.round)
    Path(train_filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(to_predict_filepath).parent.mkdir(parents=True, exist_ok=True)

    #
    if args.round == 1:
        df_train = df_train_pool[:num_train_per_round]
        df_to_predict = df_train_pool[num_train_per_round:]

        print(f"Training data shape: {df_train.shape[0]}")
        print(f"To predict data shape: {df_to_predict.shape[0]}")

        print(f"Saving training data to '{train_filepath}'")
        df_train.to_csv(train_filepath, sep='\t', index=False)

        print(f"Saving to_predict data to '{to_predict_filepath}'")
        df_to_predict.to_csv(to_predict_filepath, sep='\t', index=False)
    else:
        predicted_filepath = PREDICTED_FILEPATH_FS.format(
            args.sampling_strategy, args.run, args.round-1)
        df_predicted = pd.read_csv(predicted_filepath, sep='\t', keep_default_na=False)
        df_predicted.sort_values("prob", ascending=False, inplace=True)
        print(f"Predicted '{predicted_filepath}' size: {df_predicted.shape[0]}")

        prev_train_filepath = TRAIN_FILEPATH_FS.format(
            args.sampling_strategy, args.run, args.round-1)
        df_train = pd.read_csv(prev_train_filepath, sep='\t', keep_default_na=False)
        print(f"Previous training data size: {df_train.shape[0]}")

        if args.sampling_strategy == "certain_pos":
            df_train_new = df_predicted[:num_train_per_round].copy()
            df_to_predict = df_predicted[num_train_per_round:].copy()
        elif args.sampling_strategy == "uncertain":
            df_predicted["uncertainty"] = df_predicted["prob"].apply(lambda x: np.min([1-x, x]))
            df_predicted.sort_values("uncertainty", ascending=False, inplace=True)

            df_train_new = df_predicted[:num_train_per_round].copy()
            df_to_predict = df_predicted[num_train_per_round:].copy()

            df_train_new.drop("uncertainty", axis=1, inplace=True)
            df_to_predict.drop("uncertainty", axis=1, inplace=True)
        elif args.sampling_strategy == "stratified":
            num_each_bin = math.ceil(df_predicted.shape[0]/STRATIFIED_NUM_BINS)
            num_random_samples = math.ceil(num_train_per_round/STRATIFIED_NUM_BINS)
            print(f"Number of samples in each bin: {num_each_bin}")
            print(f"Number of samples to randomly sample from each bin: {num_random_samples}")

            train_new_list = []
            to_predict_list = []
            for x in range(STRATIFIED_NUM_BINS):
                df_subset = df_predicted.iloc[:num_each_bin].copy()
                df_predicted = df_predicted.iloc[num_each_bin:].copy()
                df_subset = df_subset.sample(frac=1, random_state=args.random_state)

                if x == (STRATIFIED_NUM_BINS - 1):
                    num_random_samples = \
                        num_train_per_round - (STRATIFIED_NUM_BINS - 1) * num_random_samples
                    print(f"Last bin size: {num_random_samples}")

                train_new_list.append(df_subset[:num_random_samples].copy())
                to_predict_list.append(df_subset[num_random_samples:].copy())

            df_train_new = pd.concat(train_new_list)
            df_to_predict = pd.concat(to_predict_list)
        else:
            raise ValueError()

        train_new_filepath = train_filepath.replace('.tsv', '_new.tsv')
        print(f"Saving training data new to this round to '{train_new_filepath}'")
        df_train_new.to_csv(train_new_filepath, sep='\t', index=False)

        df_train_new.drop("prob", axis=1, inplace=True)
        df_train = pd.concat([df_train, df_train_new])
        df_to_predict.drop("prob", axis=1, inplace=True)

        print(f"Saving training data to '{train_filepath}'")
        df_train.to_csv(train_filepath, sep='\t', index=False)

        print(f"Saving to_predict data to '{to_predict_filepath}'")
        df_to_predict.to_csv(to_predict_filepath, sep='\t', index=False)

    print(f"*****{args.sampling_strategy}|run_{args.run}|round_{args.round} END *****")


if __name__ == '__main__':
    main()
