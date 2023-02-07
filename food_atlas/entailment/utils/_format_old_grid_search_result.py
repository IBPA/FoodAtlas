import pickle

import pandas as pd


def summarize_grid_search_result(
        grid_search_result: dict) -> pd.DataFrame:
    """Summarize the grid search result.

    Args:
        grid_search_result: The result of the grid search containing all
            training and evaluation statistics.

    Returns:
        The summarized grid search result, where the first row indicates the
            hyperparameter set with the best mean validation precision score.

    """
    result_rows = []
    for k, v in grid_search_result.items():
        if k == 'failed':
            continue

        result_row = {
            'batch_size': v['hparam']['batch_size'],
            'lr': v['hparam']['lr'],
            'epochs': v['hparam']['epochs'],
            'random_seed': v['random_seed'],
            'loss': v['eval_stats']['loss'],
        }
        for k_metric, v_metric in v['eval_stats']['metrics'].items():
            result_row[k_metric] = v_metric

        result_rows += [result_row]

    result = pd.DataFrame(result_rows)
    result_summarized = result.groupby(['batch_size', 'lr', 'epochs']).mean()
    result_summarized = result_summarized.sort_values(
        'precision', ascending=False)

    return result_summarized


if __name__ == '__main__':
    with open('grid_search_all.pkl', 'rb') as f:
        grid_search_result = pickle.load(f)

    # with open('grid_search copy.pkl', 'rb') as f:
    #     grid_search_result_gt = pickle.load(f)

    # print(grid_search_result_gt)
    # print("=====================================")
    print(grid_search_result)

    formatted_all = {}
    for key, value in grid_search_result.items():
        if key == 'failed':
            continue

        formatted = {
            'hparam': {
                'batch_size': value['hparam']['batch_size'],
                'lr': value['hparam']['lr'],
                'epochs': value['hparam']['epochs']
            },
            'random_seed': value['hparam']['random_seed'],
            'train_stats': {},
            'eval_stats': None,
        }
        for k, v in value['result'].items():
            if k.startswith('train_iter'):
                formatted['train_stats'][k] = v
        formatted['eval_stats'] = value['result']['val']
        formatted_all[key] = formatted

    formatted_all['failed'] = []

    with open('grid_search_result.pkl', 'wb') as f:
        pickle.dump(formatted_all, f)

    formatted_summed = summarize_grid_search_result(formatted_all)
    formatted_summed.to_csv('grid_search_result_summary.csv')
