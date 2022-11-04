from sklearn.metrics import confusion_matrix, r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_bin_membership(probs, n_bins=5):
    probs = pd.Series(probs)
    probs_bins = pd.cut(
        probs,
        bins=[i / n_bins for i in range(n_bins + 1)],
        right=True,
        include_lowest=True,
    )
    bin_names = probs_bins.value_counts().sort_index().index.tolist()

    bin_members = {}
    for i, bin_name in enumerate(bin_names):
        bin_members[i] = probs_bins[probs_bins == bin_name].index.tolist()

    return bin_members


def get_test_prob_result(
        run_ids,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain']):
    PATH_REUSLT = "~/git/FoodAtlas/outputs/data_generation/{}/run_{}/"\
        "round_{}/test_probs.tsv"

    result_rows = []
    for al in active_learning_strategies:
        for run_id in run_ids:
            result = pd.read_csv(
                PATH_REUSLT.format(al, run_id, 10),
                sep='\t')

            result_rows += [{
                'al': al,
                'run_id': run_id,
                'labels': result['answer'].values,
                'probs': result['prob'].values,
            }]

    result = pd.DataFrame(result_rows)

    return result


def plot_reliability_diagram(
        run_ids,
        active_learning_strategies=['certain_pos', 'stratified', 'uncertain'],
        n_bins=5,
        path_save=None):
    """
    """
    result = get_test_prob_result(
        run_ids,
        active_learning_strategies,
    )

    data_plot_rows = []
    for al in active_learning_strategies:
        result_al = result[result['al'] == al]

        for row in result_al.itertuples():
            labels = row.labels
            probs = row.probs

            bin_members = get_bin_membership(probs, n_bins)
            for ib, bin_members in bin_members.items():
                labels_bin = labels[bin_members]
                probs_bin = probs[bin_members]

                tn, fp, fn, tp = confusion_matrix(
                    y_true=[1 if x == 'Entails' else 0 for x in labels_bin],
                    y_pred=[1 if x > 0.5 else 0 for x in probs_bin],
                    labels=[0, 1],
                ).ravel()

                for metric in ['pos', 'neg']:
                    data_plot_rows += [{
                        'al': al,
                        'run_id': row.run_id,
                        'bin_id': ib,
                        'metric': metric,
                        'count': tn + fp if metric == 'neg' else tp + fn,
                    }]
    data_plot = pd.DataFrame(data_plot_rows)\
        .sort_values(['al', 'run_id', 'bin_id'])

    plt.figure(figsize=(6, 6))
    ax = sns.barplot(
        data=data_plot,
        x='bin_id',
        y='count',
        hue='metric',
        hue_order=['pos', 'neg'],
        errwidth=1,
    )
    data_plot_2 = data_plot.groupby(['al', 'run_id', 'bin_id']).sum()\
        .reset_index()
    data_plot_2['pos'] = data_plot[data_plot['metric'] == 'pos']['count']\
        .reset_index(drop=True)
    data_plot_2['frac_pos'] = data_plot_2['pos'] / data_plot_2['count']
    frac_pos = data_plot_2.groupby(['bin_id']).mean()['frac_pos'].values

    bin_width = 10 // n_bins / 10
    y_true = [(x / 10) - (bin_width / 2)
              for x in range(10 // n_bins, 11, 10 // n_bins)]
    print(y_true)
    print(frac_pos)
    r2 = r2_score(
        y_true=y_true,
        y_pred=frac_pos,
    )
    ax_2 = ax.twinx()
    sns.lineplot(
        x=[0, n_bins - 1],
        y=[bin_width / 2, 1 - (bin_width / 2)],
        linestyle='--',
        color='black',
        ax=ax_2,
    )
    sns.lineplot(
        data=data_plot_2,
        x='bin_id',
        y='frac_pos',
        color='black',
        err_style='bars',
        errorbar='se',
        ax=ax_2,
    )
    ax.set_title(f"R2: {r2:.2f}")

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()
