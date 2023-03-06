import pandas as pd

from .. import FoodAtlasEntailmentModel, get_food_atlas_data_loader

if __name__ == '__main__':
    ROUNDS = [1, 2, 3]
    SEEDS = [4, 5, 1]

    for round, seed in zip(ROUNDS, SEEDS):
        model = FoodAtlasEntailmentModel(
            'biobert',
            f"/data/lfz/projects/FoodAtlas/outputs/entailment_model/{round}/"
            f"eval_best_model/seed_{seed}/model_state.pt",
        )

        data_loader = get_food_atlas_data_loader(
            path_data="outputs/data_generation/test.tsv",
            tokenizer=model.tokenizer,
            train=False,
            shuffle=False,
        )

        y_score = model.predict(data_loader)

        data_test = pd.read_csv("outputs/data_generation/test.tsv", sep='\t')
        data_test['prob'] = y_score
        data_test.to_csv(f"test_{round}_with_score.tsv", sep='\t', index=False)

        # Check accuracy.
        y_score_binary \
            = ["Entails" if p > 0.5 else "Does not entail" for p in y_score]
        data_test['prediction'] = y_score_binary
        data_test = data_test[data_test['answer'] != 'Skip']
        print(len(data_test[data_test['answer'] == data_test['prediction']]) / len(data_test))
