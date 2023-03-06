import argparse
import os
import sys

from pykeen.datasets.base import PathDataset
from pykeen.hpo import hpo_pipeline


TRAIN_DATASET_FILENAME = "train.txt"
VAL_DATASET_FILENAME = "val.txt"
TEST_DATASET_FILENAME = "test.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing train set.",
    )

    parser.add_argument(
        "--val_test_dir",
        type=str,
        required=True,
        help="Directory containing val/test set.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Models to optimize separated by commas (e.g. TransE,TransD).",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=530,
        help='Random state (Default: 530)',
    )

    return parser.parse_args()


def main():
    args = parse_argument()

    # Get a training dataset
    dataset = PathDataset(
        training_path=os.path.join(args.train_dir, TRAIN_DATASET_FILENAME),
        validation_path=os.path.join(args.val_test_dir, VAL_DATASET_FILENAME),
        testing_path=os.path.join(args.val_test_dir, TEST_DATASET_FILENAME),
        create_inverse_triples=True,
    )

    models = args.models.split(',')
    print(f'Models to optimize: {models}')

    for model in models:
        print(f'\n\n\nRunning HPO for {model}...')

        base_directory = os.path.join(args.output_dir, model)

        hpo_pipeline_result = hpo_pipeline(
            n_trials=50,
            dataset=dataset,
            model=model,
            epochs=1000,
            stopper='early',
            # save_model_directory=os.path.join(base_directory, 'models'),
        )

        hpo_pipeline_result.save_to_directory(os.path.join(base_directory, 'results'))


if __name__ == '__main__':
    main()
