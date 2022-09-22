import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open(
            # 'outputs/round_1/entailment_model/grid_search.pkl',
            'outputs/experiments/augmentation/orig/grid_search.pkl',
            'rb') as f:
        results = pickle.load(f)
    print(results)
    results_lst = []
    for k, v in results.items():
        if k == 'failed':
            continue

        hparam = v['hparam']
        val_result = v['result']['val']['metrics']

        results_lst += [{
            'batch_size': hparam['batch_size'],
            'learning_rate': hparam['lr'],
            'epochs': hparam['epochs'],
            'val_prec': val_result['precision'],
        }]

    results = pd.DataFrame(results_lst)

    # sns.barplot(
    #     x='epochs',
    #     y='val_prec',
    #     data=results,
    # )
    # plt.savefig('epochs.png')

    print(results)
