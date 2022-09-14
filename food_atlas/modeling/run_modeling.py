import torch
from transformers import AutoTokenizer
import pandas as pd
import click

from . import (
    load_model,
    train_nli_model,
    get_food_atlas_data_loader
)


def train_tuning_wrapper(
        model,
        tokenizer,
        path_data_train,
        path_data_test,
        batch_size,
        learning_rate,
        epochs):
    """
    """
    data_loader_train, data_loader_val = get_food_atlas_data_loader(
        path_data_train,
        path_data_test,
        tokenizer=tokenizer,
        batch_size=batch_size)

    train_nli_model(
        model,
        data_loader_train,
        data_loader_val=data_loader_val,
        epochs=epochs,
        lr=learning_rate,
        device='cuda')


@click.command()
@click.argument('path-data-train', type=click.Path(exists=True))
@click.argument('path-data-test', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.option('--random-seed', type=int, default=42)
def main(
        path_data_train,
        path_data_test,
        path_or_name_nli_model,
        random_seed):
    BATCH_SIZE = [16, 32]
    LEARNING_RATE = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    EPOCHS = [2, 3, 4]

    torch.manual_seed(random_seed)

    model, tokenizer = load_model(path_or_name_nli_model)
    tokenizer = AutoTokenizer.from_pretrained(path_or_name_nli_model)
    tokenizer._pad_token_type_id = 1

    # Train model.
    # config = {
    #     'batch_size': BATCH_SIZE,
    # }


if __name__ == '__main__':
    main()
