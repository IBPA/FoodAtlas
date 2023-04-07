import pandas as pd


if __name__ == '__main__':
    PATH_ROOT = "/data/lfz/projects/FoodAtlas/outputs/entailment_model/{}/" \
        "run_{}/round_10/eval_best_model/eval_results.csv"

    data_test_dfs = []
    for i_run in range(1, 101):
        for al in ['certain_pos', 'random', 'stratified', 'uncertain']:
            data = pd.read_csv(PATH_ROOT.format(al, i_run))
            data_test_df = data[data['Unnamed: 0'] == 'mean_test']
            data_test_dfs += [data_test_df]

    data_test = pd.concat(data_test_dfs, axis=0, ignore_index=True)
    data_test = data_test[data_test.columns[3:]]

    data_test['npv'] = data_test['tn'] / (data_test['tn'] + data_test['fn'])
    data_test['p'] = data_test['tp'] + data_test['fn']
    data_test['n'] = data_test['tn'] + data_test['fp']
    data_test['pn'] = data_test['tn'] + data_test['fn']
    data_test['pp'] = data_test['tp'] + data_test['fp']
    print(data_test)
    data_test_mean = data_test.mean(axis=0)
    data_test_std = data_test.std(axis=0)
    print("Average test results:")
    print(data_test_mean)
    print()
    print("STD test results:")
    print(data_test_std)
