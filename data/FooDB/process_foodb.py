import argparse
from pathlib import Path
import sys

import pandas as pd


def parse_argument() -> argparse.Namespace:
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description="Process FooDB data.")

    parser.add_argument(
        "--food_filepath",
        type=str,
        help="Food.csv filepath.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite the foods output file specified at --output_filepath.",
    )

    parser.add_argument(
        "--output_filepath",
        type=str,
        help="Output food names filepath to be used for query.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    # make output dir
    output_dir = "/".join(args.output_filepath.split("/")[:-1])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read data
    df = pd.read_csv(args.food_filepath)
    df.dropna(subset="ncbi_taxonomy_id", inplace=True)
    df = df[["name", "name_scientific"]]

    rename_dict = {
        "name": "food_name",
        "name_scientific": "food_name_synonyms",
    }
    df.rename(rename_dict, inplace=True, axis=1)
    df.drop_duplicates(inplace=True)

    df.to_csv(args.output_filepath, sep='\t', index=False)


if __name__ == '__main__':
    main()
