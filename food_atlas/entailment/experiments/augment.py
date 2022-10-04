import pandas as pd


if __name__ == '__main__':
    data = pd.read_csv('outputs/data_generation/train_1.tsv', sep='\t')

    idx_orig = [i for i in range(data.shape[0]) if i % 11 == 0]
    data_aug_0 = data.loc[idx_orig]

    idx_aug_5 = [i for i in range(data.shape[0]) if i % 11 <= 5]
    data_aug_5 = data.loc[idx_aug_5]

    data_aug_0.to_csv(
        'food_atlas/entailment/experiments/data/train_1_orig.tsv', sep='\t')
    data_aug_5.to_csv(
        'food_atlas/entailment/experiments/data/train_1_aug_5.tsv', sep='\t')

    # Use experiemtn/scripts to train models with different augmented data.
    # The conclusion shows that the augmentation does not have siginificant
    # effect on the performance. Thus, we proceed the future modeling without
    # augmentation.
