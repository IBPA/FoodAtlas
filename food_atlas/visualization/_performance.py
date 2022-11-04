import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_all_eval_results(
        run_ids=None,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain']):
    PATH_EVAL = "/data/lfz/projects/FoodAtlas/outputs/entailment_model/"\
        "{}/run_{}/round_{}/eval_best_model/eval_results.csv"

    if run_ids is None:
        run_ids = range(1, 101)

    results_dfs = []
    for al in active_learning_strategies:
        results_runs_dfs = []
        for run_id in run_ids:
            results_round_dfs = []
            for round_id in range(1, 11):
                result_round = pd.read_csv(
                    PATH_EVAL.format(al, run_id, round_id),
                    index_col=0,
                )
                result_round = result_round.loc[['mean_val', 'mean_test']]\
                    .copy()
                result_round['round_id'] = round_id
                results_round_dfs += [result_round]
            results_run = pd.concat(results_round_dfs, ignore_index=True)
            results_run['run_id'] = run_id
            results_runs_dfs += [results_run]
        results_runs = pd.concat(results_runs_dfs, ignore_index=True)
        results_runs['al'] = al
        results_dfs += [results_runs]
    results = pd.concat(results_dfs, ignore_index=True)
    results = results.query("dataset == 'test'")
    results = results[
        ['accuracy', 'precision', 'recall', 'f1', 'round_id', 'run_id', 'al']
    ]

    return results


def plot_performance_box(
        run_ids=None,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain'],
        metrics=['precision', 'recall', 'f1'],
        path_save=None):
    results = get_all_eval_results(
        run_ids=run_ids,
        active_learning_strategies=active_learning_strategies)

    _, axs = plt.subplots(len(metrics), 1, figsize=(12, 6))

    for i, metric in enumerate(metrics):
        sns.boxplot(
            data=results,
            x='round_id',
            y=metric,
            hue='al',
            ax=axs[i],
        )

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()


def plot_performance_line(
        run_ids=None,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain'],
        metrics=['precision', 'recall', 'f1'],
        path_save=None):
    results = get_all_eval_results(
        run_ids=run_ids,
        active_learning_strategies=active_learning_strategies)

    _, axs = plt.subplots(len(metrics), 1, figsize=(3, 6))

    for i, metric in enumerate(metrics):
        sns.lineplot(
            data=results,
            x='round_id',
            y=metric,
            hue='al',
            err_style='bars',
            errorbar='sd',
            ax=axs[i],
        )

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()


if __name__ == '__main__':
    metrics = ['precision', 'recall', 'f1']
    result = get_all_eval_results(
        run_ids=range(1, 101),
        active_learning_strategies=['stratified', 'uncertain'],
    )

    result_means = result.groupby(['al', 'round_id']).mean()\
        .drop('run_id', axis=1)
    result_std = result.groupby(['al', 'round_id']).std()\
        .drop('run_id', axis=1)
    result_means.columns = [f'{c}_mean' for c in result_means.columns]
    result_std.columns = [f'{c}_std' for c in result_std.columns]
    result_values = pd.concat([result_means, result_std], axis=1)
    result_values.to_csv('result_values.csv')
    print(result_values)

    from scipy.stats import ttest_ind

    # # Uncertain vs. Stratified.
    # result_stratified = result.query("al == 'stratified'")
    # result_uncertain = result.query("al == 'uncertain'")
    # for i in range(1, 1 + 10):
    #     print(f"Round {i}")
    #     result_stratified_i \
    #         = result_stratified.query("round_id == @i")[metrics]
    #     result_uncertain_i \
    #         = result_uncertain.query("round_id == @i")[metrics]

    #     for metric in metrics:
    #         t, p = ttest_ind(
    #             a=result_stratified.query("round_id == @i")[metric],
    #             b=result_uncertain.query("round_id == @i")[metric],
    #             nan_policy='omit',
    #         )
    #         print(f"    metric {metric:<10}: t={t:.4f}, p={p:.4f}")

    # # Round vs. Next round.
    # for i in range(1, 10):
    #     print(f"Round {i} vs. Round {i + 1}")

    #     for al in ['stratified', 'uncertain']:
    #         print(f"    AL {al}")
    #         result_al_i = result.query("al == @al and round_id == @i")
    #         result_al_i1 = result.query("al == @al and round_id == @i + 1")

    #         for metric in metrics:
    #             t, p = ttest_ind(
    #                 a=result_al_i[metric],
    #                 b=result_al_i1[metric],
    #                 nan_policy='omit',
    #             )
    #             print(f"        metric {metric:<10}: t={t:.4f}, p={p}")

    # Round vs. Final round.
    for i in range(1, 10):
        print(f"Round {i} vs. Round 10")

        for al in ['stratified', 'uncertain']:
            print(f"    AL {al}")
            result_al_i = result.query("al == @al and round_id == @i")
            result_al_i1 = result.query("al == @al and round_id == 10")

            for metric in metrics:
                t, p = ttest_ind(
                    a=result_al_i[metric],
                    b=result_al_i1[metric],
                    nan_policy='omit',
                )
                print(f"        metric {metric:<10}: t={t:.4f}, p={p}")
