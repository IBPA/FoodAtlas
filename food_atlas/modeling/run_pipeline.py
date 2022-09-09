import torch
from transformers import AutoTokenizer
import pandas as pd
import click

from . import (
    train_nli_model,
    get_food_atlas_data_loader
)


@click.command()
@click.argument('path-data-train', type=click.Path(exists=True))
@click.argument('path-data-test', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.option('--is-first-cycle', type=bool, default=False)
@click.option('--random-seed', type=int, default=42)
def main(
    path_data_train,
    path_data_test,
    path_or_name_nli_model,
    is_first_cycle,
    random_seed,
):
    torch.manual_seed(random_seed)

    # Load NLI model from hugging face if the first cycle. Otherwise, load from
    #   disk.
    if is_first_cycle:
        pass
    else:
        pass
    tokenizer = AutoTokenizer.from_pretrained(path_or_name_nli_model)
    tokenizer._pad_token_type_id = 1

    # Load datasets.
    data_train_loader, data_test_loader = get_food_atlas_data_loader(
        path_data_train,
        path_data_test,
        tokenizer=tokenizer,
        max_seq_len=100,
        batch_size=4,
    )

    # Train model.
    # train_nli_model()


if __name__ == '__main__':
    main()
