import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    ph_pairs_file_name = "ph_pairs_terpenedb.txt"

    data_cols = []
    for i in tqdm(range(100)):
        try:
            path = (
                f"/data/lfz/projects/FoodAtlas/outputs/entailment_model/prod/"
                f"ensemble/{i}/seed_{i}/predicted_{ph_pairs_file_name}.tsv"
            )
            data_cols += [pd.read_csv(path, sep='\t', usecols=[0])]
        except Exception:
            print(f"File {i} not found")

    data = pd.concat(data_cols, axis=1)
    data['mean'] = data.mean(axis=1)
    data['std'] = data.std(axis=1)

    data = pd.concat(
        [
            pd.read_csv(
                "outputs/data_processing/"
                f"{ph_pairs_file_name}",
                sep='\t',
            ),
            data[['mean', 'std']],
        ],
        axis=1)
    data.to_csv(
        f"outputs/data_processing/predictions_{ph_pairs_file_name}.tsv",
        sep='\t',
        index=False,
    )
