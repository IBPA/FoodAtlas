import os
import traceback
import pickle
from itertools import product

import pandas as pd
import torch
# import wandb
import click

from .. import (
    load_model,
    train,
    get_food_atlas_data_loader,
)
torch.cuda.empty_cache()


def train_tuning_wrapper(
        model,
        tokenizer,
        path_data_train,
        path_data_val,
        batch_size,
        learning_rate,
        epochs,
        verbose=True):
    """
    """
    data_loader_train = get_food_atlas_data_loader(
        path_data_train,
        tokenizer=tokenizer,
        train=True,
        batch_size=batch_size,
        shuffle=True,
        verbose=verbose)

    data_loader_val = get_food_atlas_data_loader(
        path_data_val,
        tokenizer=tokenizer,
        train=True,
        batch_size=16,
        shuffle=False,
        verbose=verbose)

    result = train(
        model,
        data_loader_train,
        data_loader_val=data_loader_val,
        epochs=epochs,
        lr=learning_rate,
        device='cuda')

    return result


@click.command()
@click.argument('path-data-train', type=click.Path(exists=True))
@click.argument('path-data-val', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.argument('path-output-dir', type=str)
@click.option('--random-seed', type=int, default=42)
def main(
        path_data_train,
        path_data_val,
        path_or_name_nli_model,
        path_output_dir,
        random_seed):
    # wandb.init(project="FoodAtlas", entity='fzli')

    os.makedirs(f"{path_output_dir}/best_f1", exist_ok=True)
    os.makedirs(f"{path_output_dir}/best_prec", exist_ok=True)
    torch.manual_seed(random_seed)

    # Train with grid search.
    results = {}
    failed = []
    # BATCH_SIZE_RANGE = [16, 32]
    # LEARNING_RATE_RANGE = [2e-5, 3e-5, 5e-5]
    # EPOCHS_RANGE = [2, 3, 4]
    BATCH_SIZE_RANGE = [32]
    LEARNING_RATE_RANGE = [3e-5]
    EPOCHS_RANGE = [2, 3, 4]
    best_f1_hparams = None
    best_f1_score = -1
    best_prec_hparams = None
    best_prec_score = -1
    for i, (batch_size, learning_rate, epochs) in enumerate(
            product(BATCH_SIZE_RANGE, LEARNING_RATE_RANGE, EPOCHS_RANGE)):
        # wandb.config = {
        #     'model': path_or_name_nli_model,
        #     'dataset': path_data_train,
        #     'batch_size': batch_size,
        #     'epochs': epochs,
        #     'lr': learning_rate,
        #     'random_seed': random_seed,
        # }
        model, tokenizer = load_model(path_or_name_nli_model)

        try:
            result = train_tuning_wrapper(
                model,
                tokenizer,
                path_data_train,
                path_data_val,
                batch_size,
                learning_rate,
                epochs,
                verbose=True if i == 0 else False)

            f1_score = result['val']['metrics']['f1']
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_f1_hparams = (batch_size, learning_rate, epochs)
                torch.save(
                    model.state_dict(),
                    f'{path_output_dir}/best_f1/model_state.pt')

            prec = result['val']['metrics']['precision']
            if prec > best_prec_score:
                # print(f'New best precision score: {prec}')
                best_prec_score = prec
                best_prec_hparams = (batch_size, learning_rate, epochs)
                torch.save(
                    model.state_dict(),
                    f'{path_output_dir}/best_prec/model_state.pt')

            results[i] = {
                'hparam': {
                    'batch_size': batch_size,
                    'lr': learning_rate,
                    'epochs': epochs,
                },
                'result': result,
            }
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(e)
            failed += [(batch_size, learning_rate, epochs, e)]
    results['failed'] = failed

    # Save results.
    with open(f'{path_output_dir}/grid_search_summary.txt', 'w') as f:
        f.write(
            f'Best F1 score: {best_f1_score} '
            f'with hparams: {best_f1_hparams}\n'
            f'Best precision score: {best_prec_score} '
            f'with hparams: {best_prec_hparams}\n'
        )
    with open(f'{path_output_dir}/grid_search.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
