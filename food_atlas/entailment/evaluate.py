import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
# import wandb
import click

from . import (
    load_model,
    evaluate,
    get_food_atlas_data_loader,
)
from .utils import get_all_metrics


@click.command()
@click.argument('path-data-test', type=click.Path(exists=True))
@click.argument('path-or-name-nli-model', type=str)
@click.argument('path-output-dir', type=str)
@click.option('--metric', type=str, default=None)
@click.option('--random-seed', type=int, default=42)
def main(
        path_data_test,
        path_or_name_nli_model,
        path_output_dir,
        metric,
        random_seed):
    torch.manual_seed(random_seed)

    model, tokenizer = load_model(path_or_name_nli_model)
    if metric is not None:
        model.load_state_dict(
            torch.load(f'{path_output_dir}/best_{metric}/model_state.pt'))

    data_loader_test = get_food_atlas_data_loader(
        path_data_test,
        tokenizer=tokenizer,
        train=False,
        batch_size=16,  # Does not matter for evaluation.
        shuffle=False)

    y_true, y_pred, y_score, loss = evaluate(
        model,
        data_loader_test,
        device='cuda')

    metrics = get_all_metrics(y_true, y_pred, y_score)
    metrics['loss'] = loss
    pd.DataFrame(metrics, index=[0]).to_csv(
        f'{path_output_dir}/best_{metric}/metrics.csv')

    # Plot performance curves.
    precsions, recalls, _ = precision_recall_curve(y_true, y_score)
    sns.lineplot(x=recalls, y=precsions)
    plt.savefig(f'{path_output_dir}/best_{metric}/pr_curve.svg')
    plt.close()

    fprs, tprs, _ = roc_curve(y_true, y_score)
    sns.lineplot(x=fprs, y=tprs)
    plt.savefig(f'{path_output_dir}/best_{metric}/roc_curve.svg')
    plt.close()


if __name__ == '__main__':
    main()
