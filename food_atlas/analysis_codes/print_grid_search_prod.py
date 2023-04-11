import pandas as pd


if __name__ == '__main__':
    best_hparams_dfs = []
    for i in range(0, 10):
        gs_summary = pd.read_csv(
            f"/data/lfz/projects/FoodAtlas/outputs/entailment_model/prod/"
            f"grid_search/fold_{i}/grid_search/grid_search_result_summary.csv"
        )
        gs_summary['fold'] = i
        best_hparams_dfs += [
            gs_summary[['fold', 'batch_size', 'lr', 'epochs', 'precision']]
        ]

    best_hparams = pd.concat(best_hparams_dfs, ignore_index=True)
    best_hparams_mean = best_hparams.groupby(
        ['batch_size', 'lr', 'epochs']
    ).mean().sort_values(
        by='precision', ascending=False
    )
    print(best_hparams_mean)
