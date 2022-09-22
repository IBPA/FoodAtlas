import pandas as pd
import torch
import click
from tqdm import tqdm

from . import (
    load_model,
    get_food_atlas_data_loader,
)


@click.command()
@click.argument('path-data-test', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.argument('path-model-state', type=str)
@click.argument('path-output', type=str)
def main(
        path_data_test,
        path_or_name_nli_model,
        path_model_state,
        path_output,
        ):
    DEVICE = 'cuda'

    model, tokenizer = load_model(path_or_name_nli_model)
    if path_model_state is not None:
        model.load_state_dict(
            torch.load(path_model_state))
    model.to(DEVICE)
    model.eval()

    data_loader_test = get_food_atlas_data_loader(
        path_data_test,
        tokenizer=tokenizer,
        train=False,
        batch_size=16,  # Does not matter for evaluation.
        shuffle=False)

    data_pred = pd.read_csv(path_data_test, sep='\t')

    y_score = []
    for (input_ids, attention_mask, token_type_ids), _ \
            in tqdm(data_loader_test):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits
        y_score += output.softmax(dim=1)[:, 1].detach().cpu().numpy().tolist()

    print(y_score)
    data_pred['prob'] = y_score
    data_pred.to_csv(
        path_output, sep='\t', index=False)


if __name__ == '__main__':
    main()
