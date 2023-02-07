import os
import traceback
import pickle
from itertools import product

import pandas as pd
import torch
import click

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


def summarize_grid_search_result(
        grid_search_result: dict) -> pd.DataFrame:
    """Summarize the grid search result.

    Args:
        grid_search_result: The result of the grid search containing all
            training and evaluation statistics.

    Returns:
        The summarized grid search result, where the first row indicates the
            hyperparameter set with the best mean validation precision score.

    """
    result_rows = []
    for k, v in grid_search_result.items():
        if k == 'failed':
            continue

        result_row = {
            'batch_size': v['hparam']['batch_size'],
            'lr': v['hparam']['lr'],
            'epochs': v['hparam']['epochs'],
            'random_seed': v['random_seed'],
            'loss': v['eval_stats']['loss'],
        }
        for k_metric, v_metric in v['eval_stats']['metrics'].items():
            result_row[k_metric] = v_metric

        result_rows += [result_row]

    result = pd.DataFrame(result_rows)
    result_summarized = result.groupby(['batch_size', 'lr', 'epochs']).mean()
    result_summarized = result_summarized.sort_values(
        'precision', ascending=False)

    return result_summarized


@click.command()
@click.argument('path-data-train', type=click.Path(exists=True))
@click.argument('path-data-val', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.argument('path-output-dir', type=str)
@click.option('--path-model-state', type=str, default=None)
@click.option(
    '--batch-sizes', type=str, default='16,32',
    callback=lambda ctx, param, value: [int(x) for x in value.split(',')])
@click.option(
    '--learning-rates', type=str, default='2e-5,3e-5,5e-5',
    callback=lambda ctx, param, value: [float(x) for x in value.split(',')])
@click.option(
    '--nums-epochs', type=str, default='2,3,4',
    callback=lambda ctx, param, value: [int(x) for x in value.split(',')])
@click.option(
    '--seeds', type=str, default='1,2,3,4,5',
    callback=lambda ctx, param, value: [int(x) for x in value.split(',')])
def main(
        path_data_train,
        path_data_val,
        path_or_name_nli_model,
        path_output_dir,
        path_model_state,
        batch_sizes,
        learning_rates,
        nums_epochs,
        seeds):
    os.makedirs(f"{path_output_dir}", exist_ok=True)

    # Train with the grid search, optimizing the precision score.
    grid_search_result = {}
    failed = []  # Catch the failed runs.
    for i, (batch_size, learning_rate, epochs, random_seed) in enumerate(
            product(batch_sizes, learning_rates, nums_epochs, seeds)):
        torch.manual_seed(random_seed)

        # Load a pre-trained model.
        model = FoodAtlasEntailmentModel(
            path_or_name=path_or_name_nli_model,
            path_model_state=path_model_state,
        )

        try:
            train_stats, eval_stats = train_tuning_wrapper(
                model,
                path_data_train,
                path_data_val,
                batch_size,
                learning_rate,
                epochs,
                verbose=True if i == 0 else False)

            y_true_eval, y_pred_eval, y_score_eval, loss_eval = eval_stats
            metrics_eval = get_all_metrics(
                y_true_eval, y_pred_eval, y_score_eval
            )

            grid_search_result[i] = {
                'hparam': {
                    'batch_size': batch_size,
                    'lr': learning_rate,
                    'epochs': epochs,
                },
                'random_seed': random_seed,
                'train_stats': train_stats,
                'eval_stats': {
                    'metrics': metrics_eval,
                    'loss': loss_eval,
                },
            }
        except Exception as e:  # Catch the failed runs.
            traceback.print_tb(e.__traceback__)
            print(e)
            failed += [(batch_size, learning_rate, epochs, e)]
    grid_search_result['failed'] = failed

    # Save grid search results.
    with open(f'{path_output_dir}/grid_search_result.pkl', 'wb') as f:
        pickle.dump(grid_search_result, f)

    # Save the summary of the results.
    result_summarized = summarize_grid_search_result(grid_search_result)
    result_summarized.to_csv(
        f'{path_output_dir}/grid_search_result_summary.csv')


if __name__ == '__main__':
    main()
