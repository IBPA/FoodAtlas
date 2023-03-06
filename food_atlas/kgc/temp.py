import argparse
import os
import sys

from pykeen.models import TransE
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import ClassificationEvaluator
from pykeen.datasets.base import PathDataset
from pykeen.stoppers import EarlyStopper
from pykeen.hpo import hpo_pipeline


TRAIN_DATASET_FILENAME = "train.txt"
VAL_DATASET_FILENAME = "val.txt"
TEST_DATASET_FILENAME = "test.txt"


def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset directory.",
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
        training_path=os.path.join(args.dataset_dir, TRAIN_DATASET_FILENAME),
        validation_path=os.path.join(args.dataset_dir, VAL_DATASET_FILENAME),
        testing_path=os.path.join(args.dataset_dir, TEST_DATASET_FILENAME),
        create_inverse_triples=True,
    )

    evaluation_kwargs = dict(
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ]
    )

    hpo_pipeline_result = hpo_pipeline(
        n_trials=1,
        dataset=dataset,
        model='TransE',
        evaluator='ClassificationEvaluator',
        # metric='f1_score',
        evaluation_kwargs=evaluation_kwargs,
        epochs=10,
        stopper='early',
        stopper_kwargs=dict(frequency=10, patience=5, relative_delta=0.002, metric='f1_score'),
    )
    hpo_pipeline_result.save_to_directory('../../outputs/kgc/pykeen/test')


if __name__ == '__main__':
    main()
