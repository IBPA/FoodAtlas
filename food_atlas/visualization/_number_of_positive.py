import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_positive_result(
        run_ids,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain']):
    """
    """
    PATH_REUSLT = "~/git/FoodAtlas/outputs/data_generation/{}/run_{}/"\
        "round_{}/train.tsv"

    result_rows = []
    for al in active_learning_strategies:
        for run_id in run_ids:
            for round_id in range(1, 11):
                result = pd.read_csv(
                    PATH_REUSLT.format(al, run_id, round_id),
                    sep='\t')
                n_pos = len(result.query("answer == 'Entails'"))
                result_rows += [{
                    'al': al,
                    'run_id': run_id,
                    'round_id': round_id,
                    'n_pos': n_pos,
                }]

    return pd.DataFrame(result_rows)


def plot_number_of_positive(
        run_ids,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain'],
        path_save=None):
    """
    """
    result = get_positive_result(
        run_ids,
        active_learning_strategies)

    plt.figure(figsize=(6, 6))
    sns.lineplot(
        data=result,
        x='round_id',
        y='n_pos',
        hue='al',
        err_style='bars',
        errorbar='sd',
    )

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()


if __name__ == '__main__':
    active_learning_strategies = ['stratified', 'uncertain']

    data = get_positive_result(
        run_ids=range(1, 101),
        active_learning_strategies=active_learning_strategies,
    )

    n_poss_rows = []
    for al in ['stratified', 'uncertain']:
        data_al = data.query("al == @al")
        data_al_dict = data_al.groupby('round_id').mean()['n_pos'].to_dict()
        data_al_dict['al'] = al
        n_poss_rows += [data_al_dict]
    n_poss = pd.DataFrame(n_poss_rows).set_index('al')
    print(n_poss)

    n_poss_diff = n_poss.loc['uncertain'] - n_poss.loc['stratified']
    n_poss_diff_ratio = n_poss_diff / n_poss.loc['stratified']
    print(n_poss_diff)
    print(n_poss_diff_ratio)

    print(n_poss_diff.mean(), n_poss_diff.std())
    print(n_poss_diff_ratio.mean(), n_poss_diff_ratio.std())
