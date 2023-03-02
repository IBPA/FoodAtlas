import argparse
import os
import sys

from pykeen.models import TransE
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import ClassificationEvaluator
from pykeen.datasets.base import PathDataset
from pykeen.stoppers import EarlyStopper

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

    # Pick a model
    model = TransE(triples_factory=dataset.training)
    print(model.hpo_default)

    # Pick an optimizer from Torch
    optimizer = Adam(params=model.get_grad_params())

    # Pick a training approach (sLCWA or LCWA)
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=dataset.training,
        optimizer=optimizer,
    )

    # Evaluator
    evaluator = ClassificationEvaluator()

    # Early stopper
    stopper = EarlyStopper(
        model=model,
        evaluator=evaluator,
        training_triples_factory=dataset.training,
        evaluation_triples_factory=dataset.validation,
        frequency=10,
        patience=3,
        relative_delta=0.002,
        metric='f1_score',
    )

    # Train like Cristiano Ronaldo
    _ = training_loop.train(
        triples_factory=dataset.training,
        num_epochs=1000,
        batch_size=256,
        stopper=stopper,
    )

    # Pick an evaluator
    evaluator = ClassificationEvaluator()

    # Get triples to test
    mapped_triples = dataset.testing.mapped_triples

    # Evaluate
    results = evaluator.evaluate(
        model=model,
        mapped_triples=mapped_triples,
        batch_size=1024,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    print(results.get_metric('f1_score'))


if __name__ == '__main__':
    main()
