import pprint

import pandas as pd

from .. import get_food_atlas_data_loader, FoodAtlasEntailmentModel
from ..utils import get_all_metrics


if __name__ == '__main__':
    # Check class distribution per dataset.
    ROUNDS = [1, 2]
    BEST_SEEDS = [4, 5]
    result_rows = []

    # Best R1 Model.
    for round, seed in zip(ROUNDS, BEST_SEEDS):
        model = FoodAtlasEntailmentModel(
            'biobert',
            f"/data/lfz/projects/FoodAtlas/outputs/entailment_model/{round}/"
            f"eval_best_model/seed_{seed}/model_state.pt",
        )
        data_loader_test = get_food_atlas_data_loader(
            path_data='outputs/data_generation/test.tsv',
            tokenizer=model.tokenizer,
            train=False,
            shuffle=False,
        )
        y_score = model.predict(data_loader_test)
        y_pred = ['Entails' if p > 0.5 else 'Does not entail' for p in y_score]
        data_test = pd.read_csv('outputs/data_generation/test.tsv', sep='\t')
        data_test['prediction'] = y_pred
        data_test = data_test[data_test['answer'] != 'Skip']
        data_test['answer'] = data_test['answer'].map(
            {'Entails': 1, 'Does not entail': 0}
        )
        data_test['prediction'] = data_test['prediction'].map(
            {'Entails': 1, 'Does not entail': 0}
        )
        data_test_part = data_test[
            data_test['head'].str.contains('organism_with_part')
        ]
        data_test_whole = data_test[
            ~data_test['head'].str.contains('organism_with_part')
        ]

        datasets = [data_test, data_test_part, data_test_whole]
        dataset_names = ['all', 'part', 'whole']
        for data, name in zip(datasets, dataset_names):
            # Print metrics.
            result_row = get_all_metrics(
                data['answer'],
                data['prediction'],
            )
            result_row['dataset'] = name
            result_row['round'] = round
            result_rows += [result_row]

    print(pd.DataFrame(result_rows))
