import argparse
from collections import Counter
from pathlib import Path
import random
import sys

sys.path.append('..')

from common_utils.utils import read_annotated  # noqa: E402


VAL_POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/val_post_annotation.tsv"
TEST_POST_ANNOTATION_FILEPATH = "../../outputs/data_generation/test_post_annotation.tsv"
VAL_FILEPATH = "../../outputs/data_generation/val.tsv"
TEST_FILEPATH = "../../outputs/data_generation/test.tsv"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first version of annotation.")

    parser.add_argument(
        "--random_state",
        type=int,
        help="Set random state.",
    )

    parser.add_argument(
        "--train_post_annotation_filepath",
        type=str,
        required=True,
        help="Annotation filepath.",
    )

    parser.add_argument(
        "--val_post_annotation_filepath",
        type=str,
        default=VAL_POST_ANNOTATION_FILEPATH,
        help=f"Validation set post-annotation filepath. (Default: {VAL_POST_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--test_post_annotation_filepath",
        type=str,
        default=TEST_POST_ANNOTATION_FILEPATH,
        help=f"Test set post-annotation filepath. (Default: {TEST_POST_ANNOTATION_FILEPATH})",
    )

    parser.add_argument(
        "--train_filepath",
        type=str,
        required=True,
        help="Train filepath.",
    )

    parser.add_argument(
        "--val_filepath",
        type=str,
        default=VAL_FILEPATH,
        help=f"Validation set final processed filepath. (Default: {VAL_FILEPATH})",
    )

    parser.add_argument(
        "--test_filepath",
        type=str,
        default=TEST_FILEPATH,
        help=f"Test set final processed filepath. (Default: {TEST_FILEPATH})",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()

    if args.random_state:
        random.seed(args.random_state)

    df_train = read_annotated(args.train_post_annotation_filepath)
    df_train = df_train[df_train["answer"] != "Skip"]
    df_train = df_train.sample(frac=1)
    print("Train shape before dropping duplicates by hypothesis_string: ", df_train.shape)
    df_train.drop_duplicates("hypothesis_string", inplace=True)
    print("Train shape after dropping duplicates by hypothesis_string: ", df_train.shape)
    print("Train distribution: ", Counter(df_train["answer"].tolist()))
    Path(args.train_filepath).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving training to: {args.train_filepath}")
    df_train.to_csv(args.train_filepath, sep='\t', index=False)

    df_val = read_annotated(args.val_post_annotation_filepath)
    df_val = df_val[df_val["answer"] != "Skip"]
    df_val = df_val.sample(frac=1)
    print("Val shape before dropping duplicates by hypothesis_string: ", df_val.shape)
    df_val.drop_duplicates("hypothesis_string", inplace=True)
    print("Val shape after dropping duplicates by hypothesis_string: ", df_val.shape)
    print("Val distribution: ", Counter(df_val["answer"].tolist()))
    Path(args.val_filepath).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving validation to: {args.val_filepath}")
    df_val.to_csv(args.val_filepath, sep='\t', index=False)

    df_test = read_annotated(args.test_post_annotation_filepath)
    df_test = df_test[df_test["answer"] != "Skip"]
    df_test = df_test.sample(frac=1)
    print("Test shape before dropping duplicates by hypothesis_string: ", df_test.shape)
    df_test.drop_duplicates("hypothesis_string", inplace=True)
    print("Test shape after dropping duplicates by hypothesis_string: ", df_test.shape)
    print("Test distribution: ", Counter(df_test["answer"].tolist()))
    Path(args.test_filepath).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving test to: {args.test_filepath}")
    df_test.to_csv(args.test_filepath, sep='\t', index=False)

    all_size = df_train.shape[0] + df_val.shape[0] + df_test.shape[0]
    train_ratio = df_train.shape[0]/all_size
    val_ratio = df_val.shape[0]/all_size
    test_ratio = df_test.shape[0]/all_size
    print(f"train:val:test = {train_ratio:.2f}:{val_ratio:.2f}:{test_ratio:.2f}")


if __name__ == '__main__':
    main()
