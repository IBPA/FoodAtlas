import os
import pickle
import traceback

import pandas as pd
import click
import torch

from . import (
    FoodAtlasEntailmentModel,
    get_food_atlas_data_loader,
)
from .utils import get_all_metrics


def train_tuning_wrapper(
        model: FoodAtlasEntailmentModel,
        path_data_train: str,
        path_data_val: str,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        verbose=True) -> tuple:
    """A wrapper of training function for tuning hyperparameters.

    Args:
        model: A model to be trained.
        path_data_train: Path to the training data.
        path_data_val: Path to the validation data.
        batch_size: Batch size.
        learning_rate: Learning rate.
        epochs: Number of epochs.
        verbose: Whether to print out the training progress.

    Returns:
        A tuple of training and evaluation statistics.

    """
    data_loader_train = get_food_atlas_data_loader(
        path_data_train,
        tokenizer=model.tokenizer,
        train=True,
        batch_size=batch_size,
        shuffle=True,
        verbose=verbose)
    train_stats = model.train(
        data_loader_train,
        epochs=epochs,
        lr=learning_rate)

    data_loader_val = get_food_atlas_data_loader(
        path_data_val,
        tokenizer=model.tokenizer,
        train=True,
        batch_size=4,  # Small batch size for validation to avoid OOM.
        shuffle=False,
        verbose=verbose)
    eval_stats = model.evaluate(data_loader_val)

    return train_stats, eval_stats


@click.command()
@click.argument('path-data-train', type=click.Path(exists=True))
@click.argument('path-data-val', type=click.Path(exists=True))
@click.argument('path-data-test', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.argument('path-output-dir', type=str)
def main(
        path_data_train,
        path_data_val,
        path_data_test,
        path_or_name_nli_model,
        path_output_dir):
    # Define the number of runs.
    RANDOM_SEED_RANGE = [1, 2, 3, 4, 5]

    # Load the hyperparameters of the best model.
    batch_size, lr, epochs = pd.read_csv(
        f"{path_output_dir}/grid_search_result_summary.csv"
    ).iloc[0][['batch_size', 'lr', 'epochs']]
    batch_size = int(batch_size)
    epochs = int(epochs)
    print(
        f"Best hyperparameters: batch_size={batch_size}, lr={lr}, "
        f"epochs={epochs}"
    )

    failed = []
    for seed in RANDOM_SEED_RANGE:
        torch.manual_seed(seed)
        os.makedirs(f"{path_output_dir}/seed_{seed}", exist_ok=True)

        model = FoodAtlasEntailmentModel(
            path_or_name=path_or_name_nli_model
        )

        try:
            # Get the training and validation statistics.
            train_stats, eval_stats = train_tuning_wrapper(
                model=model,
                path_data_train=path_data_train,
                path_data_val=path_data_val,
                batch_size=batch_size,
                learning_rate=lr,
                epochs=epochs,
            )
            y_true_eval, y_pred_eval, y_score_eval, loss_eval = eval_stats
            metrics_eval = get_all_metrics(
                y_true_eval, y_pred_eval, y_score_eval
            )

            # Get the test statistics.
            data_loader_test = get_food_atlas_data_loader(
                path_data_test,
                tokenizer=model.tokenizer,
                train=True,
                batch_size=4,  # Small batch size for validation to avoid OOM.
                shuffle=False)
            test_stats = model.evaluate(data_loader_test)
            y_true_test, y_pred_test, y_score_test, loss_test = test_stats
            metrics_test = get_all_metrics(
                y_true_test, y_pred_test, y_score_test
            )

            # Store the statistics and the model.
            result = {
                'train_stats': train_stats,
                'eval_stats': {
                    'metrics': metrics_eval,
                    'loss': loss_eval,
                },
                'test_stats': {
                    'metrics': metrics_test,
                    'loss': loss_test,
                },
            }
            with open(
                    f"{path_output_dir}/seed_{seed}/result.pkl",
                    'wb') as f:
                pickle.dump(result, f)
            model.save_model(
                f"{path_output_dir}/seed_{seed}/model_state.pt"
            )

        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e)
            failed += [(seed, e.__traceback__, e)]

    if failed:
        print(f"Failed: {failed}")


if __name__ == '__main__':
    main()
