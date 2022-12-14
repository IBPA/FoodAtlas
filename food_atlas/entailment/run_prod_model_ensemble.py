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


@click.command()
@click.argument('path-data-train', type=click.Path(exists=True))
@click.argument('path-data-to-predict', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.argument('path-output-dir', type=str)
@click.option(
    '--batch-size', type=int, default=16)
@click.option(
    '--learning-rate', type=float, default=5e-5)
@click.option(
    '--nums-epoch', type=int, default=4)
@click.option(
    '--seeds', type=str,
    default=','.join([str(x) for x in list(range(1, 101))]),
    callback=lambda ctx, param, value: [int(x) for x in value.split(',')])
def main(
        path_data_train,
        path_data_to_predict,
        path_or_name_nli_model,
        path_output_dir,
        batch_size,
        learning_rate,
        nums_epoch,
        seeds
        ):
    # Load the hyperparameters of the best model.
    failed = []
    for seed in seeds:
        torch.manual_seed(seed)
        os.makedirs(f"{path_output_dir}/seed_{seed}", exist_ok=True)

        model = FoodAtlasEntailmentModel(
            path_or_name=path_or_name_nli_model
        )

        try:
            # Get the training.
            data_loader_train = get_food_atlas_data_loader(
                path_data_train,
                tokenizer=model.tokenizer,
                train=True,
                batch_size=batch_size,
                shuffle=True)
            train_stats = model.train(
                data_loader_train,
                epochs=nums_epoch,
                lr=learning_rate)

            # Predict the unlabelled data.
            data_loader_pred = get_food_atlas_data_loader(
                path_data_to_predict,
                tokenizer=model.tokenizer,
                train=False,
                batch_size=4,  # Small batch size for validation to avoid OOM.
                shuffle=False)
            y_score = model.predict(data_loader_pred)

            # Save everything.
            # Save training statistics.
            result = {'train_stats': train_stats}
            with open(
                    f"{path_output_dir}/seed_{seed}/result_train.pkl",
                    'wb') as f:
                pickle.dump(result, f)
            # Save the model.
            model.save_model(
                f"{path_output_dir}/seed_{seed}/model_state.pt"
            )
            # Save the predicted probabilities.
            data_pred = pd.read_csv(path_data_to_predict, sep='\t')
            data_pred['prob'] = y_score
            data_pred.to_csv(
                f"{path_output_dir}/seed_{seed}/predicted.tsv",
                sep='\t',
                index=False)

        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e)
            failed += [(seed, e.__traceback__, e)]

    if failed:
        print(f"Failed: {failed}")


if __name__ == '__main__':
    main()
