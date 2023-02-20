import time

import pandas as pd
import click

from . import (
    FoodAtlasEntailmentModel,
    get_food_atlas_data_loader,
)


@click.command()
@click.argument('path-data-to-predict', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.argument('path-model-state', type=str)
@click.option(
    '--path-output-data-to-predict',
    type=str,
    default=f"predicted_{time.strftime('%Y%m%d-%H%M%S')}.tsv",
)
@click.option(
    '--append', type=bool, default=False
)
def main(
        path_data_to_predict,
        path_or_name_nli_model,
        path_model_state,
        path_output_data_to_predict,
        append,
        ):
    model = FoodAtlasEntailmentModel(
        path_or_name_nli_model,
        path_model_state=path_model_state,
    )

    data_loader = get_food_atlas_data_loader(
        path_data_to_predict,
        tokenizer=model.tokenizer,
        train=False,
        batch_size=4,  # Keep it small to avoid OOM.
        shuffle=False)

    y_score = model.predict(data_loader)
    if append:
        data_pred = pd.read_csv(path_data_to_predict, sep='\t')
        data_pred['prob'] = y_score
        data_pred.to_csv(
            path_output_data_to_predict, sep='\t', index=False)
    else:
        pd.DataFrame(y_score).to_csv(
            path_output_data_to_predict, sep='\t', index=False)


if __name__ == '__main__':
    main()
