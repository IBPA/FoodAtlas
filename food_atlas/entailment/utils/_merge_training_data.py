import pandas as pd


if __name__ == '__main__':
    PATH_DATA = "outputs/data_generation"

    data_1 = pd.read_csv(f"{PATH_DATA}/1/train_1.tsv", sep='\t')
    data_2 = pd.read_csv(f"{PATH_DATA}/2/random_sample_each_bin/train_2.tsv", sep='\t')

    print(data_1)
    print(data_2)

    print(data_2['answer'].value_counts())

    data = pd.concat([data_1, data_2], ignore_index=True)
    print(data['answer'].value_counts())

    data.to_csv(f"tests/train_1_2.tsv", sep='\t', index=False)
