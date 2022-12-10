import argparse
import random

import numpy as np
import pandas as pd


INPUT_FILEPATH = "../../outputs/data_generation/{}/run_{}/round_{}/to_predict.tsv"
OUTPUT_FILEPATH = "../../outputs/data_generation/{}/run_{}/round_{}/predicted.tsv"


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
        help="Sampling strategy to use.",
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

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    if args.random_state:
        random.seed(args.random_state)
        np.random.seed(args.random_state)

    input_filepath = INPUT_FILEPATH.format(args.sampling_strategy, args.run, args.round)
    output_filepath = OUTPUT_FILEPATH.format(args.sampling_strategy, args.run, args.round)

    df = pd.read_csv(input_filepath, sep='\t', keep_default_na=False)
    prob = [np.random.normal(loc=0.05, scale=0.2) for _ in range(df.shape[0])] + \
           [np.random.normal(loc=0.95, scale=0.2) for _ in range(df.shape[0])]
    prob = [np.clip(x, 0, 1) for x in prob]
    random.shuffle(prob)
    df["prob"] = prob[:df.shape[0]]
    df.to_csv(output_filepath, sep='\t', index=False)


if __name__ == '__main__':
    main()
