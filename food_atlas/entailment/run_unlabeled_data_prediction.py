import os
import time
import pickle
import traceback

import pandas as pd
from sklearn.metrics import precision_score
import click
import torch

from . import (
    FoodAtlasEntailmentModel,
    get_food_atlas_data_loader,
)
from .utils import get_all_metrics


def get_best_seed(
        path_output_dir: str,
        seeds: list[int] = [1, 2, 3, 4, 5]) -> tuple:
    """Return the best seed based on the validation set.

    Args:
        path_output_dir: Path to the output directory.
        seeds: List of seeds.

    Returns:
        int: The best seed.
        float: The best validation precision.

    """
    best_seed = None
    best_prec = -1
    for seed in seeds:
        path_result = f"{path_output_dir}/seed_{seed}/result.pkl"

        with open(path_result, 'rb') as f:
            result = pickle.load(f)
        curr_prec = result['eval_stats']['metrics']['precision']

        if curr_prec > best_prec:
            best_prec = curr_prec
            best_seed = seed

    return best_seed, best_prec


@click.command()
@click.argument('path-data-to-predict', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.argument('path-output-dir', type=str)
@click.option(
    '--path-output-data-to-predict',
    type=str,
    default=f"{os.path.abspath(os.path.dirname(__file__))}/../../outputs/"
    f"data_generation/to_predict_{time.strftime('%Y%m%d-%H%M%S')}"
    "_predicted.tsv"
)
def main(
        path_data_to_predict,
        path_or_name_nli_model,
        path_output_dir,
        path_output_data_to_predict):
    # Define the number of runs.
    RANDOM_SEED_RANGE = [1, 2, 3, 4, 5]

    best_seed, best_prec = get_best_seed(
        path_output_dir, seeds=RANDOM_SEED_RANGE
    )
    print(
        f"best seed: {best_seed}, best precision: {best_prec:.4f}"
    )

    model = FoodAtlasEntailmentModel(
        path_or_name_nli_model,
        path_model_state=f"{path_output_dir}/seed_{best_seed}/model_state.pt",
    )

    data_loader = get_food_atlas_data_loader(
        path_data_to_predict,
        tokenizer=model.tokenizer,
        train=False,
        batch_size=4,  # Keep it small to avoid OOM.
        shuffle=False)

    y_score = model.predict(data_loader)
    data_pred = pd.read_csv(path_data_to_predict, sep='\t')
    data_pred['prob'] = y_score
    data_pred.to_csv(
        path_output_data_to_predict, sep='\t', index=False)


if __name__ == '__main__':
    main()
