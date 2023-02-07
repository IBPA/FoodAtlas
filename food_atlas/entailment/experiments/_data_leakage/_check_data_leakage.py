import pandas as pd

from ... import FoodAtlasEntailmentModel, get_food_atlas_data_loader

if __name__ == '__main__':
    # ROUNDS = [1, 2, 3]
    # SEEDS = [4, 5, 1]
    # ROUNDS = [1]
    # SEEDS = [5]

    # for round, seed in zip(ROUNDS, SEEDS):
    #     model = FoodAtlasEntailmentModel(
    #         'biobert',
    #         f"/data/lfz/projects/FoodAtlas/outputs/entailment_model/{round}/"
    #         f"eval_best_model/seed_{seed}/model_state.pt",
    #     )

    #     data_loader = get_food_atlas_data_loader(
    #         # path_data="outputs/data_generation/test.tsv",
    #         path_data=f"outputs/data_generation/{round}/to_predict_{round}.tsv",
    #         tokenizer=model.tokenizer,
    #         train=False,
    #         shuffle=False,
    #     )

    #     y_score = model.predict(data_loader)

    #     data_test = pd.read_csv(
    #         f"outputs/data_generation/{round}/to_predict_{round}.tsv",
    #         sep='\t')
    #     data_test['prob'] = y_score
    #     data_test.to_csv(
    #         f"predicted_{round}_with_score.tsv", sep='\t', index=False)

    #     # Check accuracy.
    #     y_score_binary \
    #         = ["Entails" if p > 0.5 else "Does not entail" for p in y_score]
    #     data_test['prediction'] = y_score_binary
    #     data_test = data_test[data_test['answer'] != 'Skip']
    #     print(len(data_test[data_test['answer'] == data_test['prediction']]) / len(data_test))

    import sys
    import numpy as np
    import pandas as pd
    from scipy.stats import ttest_ind
    train_files = [
        "outputs/data_generation/1/train_1.tsv",
        # "../../outputs/data_generation/2/random_sample_each_bin/train_2.tsv",
        # "../../outputs/data_generation/3/random_sample_each_bin/train_3.tsv",
    ]
    data = []
    for x in train_files:
        df = pd.read_csv(x, sep='\t', keep_default_na=False)
        data.append(df)
    df_train = pd.concat(data)
    train_hypotheses = list(set(df_train["hypothesis_string"].tolist()))
    df_test = pd.read_csv(
        "/home/jasonyoun/Jason/Scratch/temp/test_1_with_score.tsv",
        sep='\t',
        keep_default_na=False,
    )
    test_hypotheses = list(set(df_test["hypothesis_string"].tolist()))
    train_overlap = [x for x in train_hypotheses if x in test_hypotheses]
    print(len(train_overlap))
    overlapping = []
    non_overlapping = []
    for idx, row in df_test.iterrows():
        if row["hypothesis_string"] in train_hypotheses:
            overlapping.append(row["prob"])
        else:
            non_overlapping.append(row["prob"])
    print(len(overlapping))
    print(len(non_overlapping))
    print(np.mean(overlapping))
    print(np.mean(non_overlapping))
    print(ttest_ind(overlapping, non_overlapping, equal_var=False))

    data_test = pd.read_csv(
        f"test_1_with_score.tsv", sep='\t')
    print(data_test)
