import os

import pandas as pd
import click


@click.command()
@click.argument(
    'path-input-dir',
    type=click.Path(exists=True),
    default="outputs/data_generation",
)
@click.argument(
    'path-output-dir',
    type=click.Path(),
)
@click.argument(
    'num-round',
    type=int,
)
@click.argument(
    'al-strategy',
    type=str,
    default='random_sample_each_bin',
)
def main(
        path_input_dir,
        path_output_dir,
        num_round,
        al_strategy,
        ):
    if al_strategy not in ['random_sample_each_bin']:
        raise ValueError(
            f"active learning strategy {al_strategy} not supported. "
            "It must be one of ['random_sample_each_bin']"
        )
    os.makedirs(path_output_dir, exist_ok=True)

    data_list = []
    for i in range(1, num_round + 1):
        if i == 1:
            path = f"{path_input_dir}/{i}/train_{i}.tsv"
        else:
            path = f"{path_input_dir}/{i}/{al_strategy}/train_{i}.tsv"

        data = pd.read_csv(path, sep='\t')
        data_list += [data]

    data = pd.concat(data_list, ignore_index=True)
    data.to_csv(
        f"{path_output_dir}/train_{num_round}_merged.tsv",
        sep='\t',
        index=False)


if __name__ == '__main__':
    main()
